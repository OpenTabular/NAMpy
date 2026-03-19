import numpy as np

from ..base import (
    BaseSmoothTerm,
    _resolve_feature,
    _resolve_numeric_by,
    _is_effectively_constant,
)
from ..registry import register_smooth
from ...design.objects import PenaltyDefinition
from ...design.penalties import null_space_penalty_from_penalty
from ....splines.univariate_bases import (
    pspline_knots,
    pspline_difference_penalty,
    pspline_predict_matrix,
)


@register_smooth("ps")
class PSplineTerm1D(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "ps"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=10,
        basis="ps",
        m=None,
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        constraint_mode="auto",
        pc=None,
        knots=None,
        null_penalty_tol=1e-10,
        metadata=None,
    ):
        super().__init__(
            feature=feature,
            label=label,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = int(k)
        self.basis_name = str(basis).lower()
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.null_penalty_tol = float(null_penalty_tol)

        if m is None:
            self.m = (2, 2)
        elif np.isscalar(m):
            self.m = (int(m), int(m))
        else:
            vals = tuple(int(v) for v in m)
            if len(vals) == 1:
                self.m = (vals[0], vals[0])
            elif len(vals) == 2:
                self.m = vals
            else:
                raise ValueError("For bs='ps', m must have length 1 or 2.")

        if self.basis_name != "ps":
            raise NotImplementedError(
                f"PSplineTerm1D currently supports only basis='ps', got {basis!r}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_index = None
        self._feature_name = None
        self._by_index = None
        self._by_name = None
        self._by_values = None
        self._by_is_constant = None

        self._basis_train = None
        self._penalties = None
        self._ps_knots = None

    def fit(self, X, feature_names):
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name

        xj = np.asarray(X[:, idx], dtype=np.float64)

        if self.by is not None:
            self._by_index, self._by_name, self._by_values = _resolve_numeric_by(
                self.by, X, feature_names
            )
            self._by_is_constant = _is_effectively_constant(self._by_values)
        else:
            self._by_index, self._by_name, self._by_values = None, None, None
            self._by_is_constant = True

        if self.constraint_mode == "factor_by":
            raise NotImplementedError(
                "factor-by replicated P-splines are not yet implemented."
            )
        if self.pc is not None:
            raise NotImplementedError("pc=... is not yet implemented for bs='ps'.")

        basis_order, penalty_order = self.m
        if basis_order < 0 or penalty_order < 0:
            raise ValueError("For bs='ps', m entries must be >= 0.")

        self._ps_knots = pspline_knots(
            xj,
            bs_dim=self.k,
            basis_order=basis_order,
            supplied_knots=self.knots,
        )

        degree = basis_order + 1
        from ....splines.univariate_bases import bspline_design_matrix

        base = bspline_design_matrix(
            xj,
            self._ps_knots,
            degree=degree,
            deriv=0,
            extrapolate=True,
        )

        if self._by_values is not None:
            base = base * self._by_values[:, None]

        self._basis_train = np.asarray(base, dtype=np.float64)

        if self.fixed:
            self._penalties = []
        else:
            S = pspline_difference_penalty(self._basis_train.shape[1], penalty_order)
            self._penalties = [0.5 * (S + S.T)]

        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        if len(self.penalties) == 0:
            return []

        if self.select and self.sp is not None:
            raise NotImplementedError(
                "term-level sp is not yet implemented for select=True smooths in the "
                "current runtime, because select adds an extra null-space penalty."
            )

        main_penalty = np.asarray(self.penalties[0], dtype=np.float64)
        sp_vals = self._normalized_term_sp(1)
        sp_main = sp_vals[0] if sp_vals else None

        if sp_main is None:
            sp_mode = None
            sp_value = None
        elif sp_main >= 0:
            sp_mode = "fixed"
            sp_value = float(sp_main)
        else:
            sp_mode = "estimate"
            sp_value = None

        defs = [
            PenaltyDefinition(
                matrix=main_penalty,
                smoothing_id=(None if self.smoothing_id is None else str(self.smoothing_id)),
                kind="smooth",
                sp_mode=sp_mode,
                sp_value=sp_value,
                metadata={
                    "term_type": self.term_type,
                    "basis_name": self.basis_name,
                    "feature": self.feature,
                    "label": self.label,
                    "by": self.by,
                    "by_name": self._by_name,
                    "by_is_constant": bool(self._by_is_constant),
                    "constraint_mode": self.constraint_mode,
                    "pc": self.pc,
                    "knots": self.knots,
                    "m": self.m,
                    "fixed": bool(self.fixed),
                    "term_sp": sp_main,
                    "is_selection_penalty": False,
                },
            )
        ]

        if self.select:
            S0, meta = null_space_penalty_from_penalty(
                main_penalty,
                tol=self.null_penalty_tol,
            )
            if meta["rank"] > 0:
                defs.append(
                    PenaltyDefinition(
                        matrix=S0,
                        smoothing_id=(
                            None
                            if self.smoothing_id is None
                            else f"{self.smoothing_id}::select"
                        ),
                        kind="null_space",
                        rank=meta["rank"],
                        null_space_dim=meta["null_space_dim"],
                        is_null_space_penalty=True,
                        sp_mode=None,
                        sp_value=None,
                        metadata={
                            "term_type": self.term_type,
                            "basis_name": self.basis_name,
                            "feature": self.feature,
                            "label": self.label,
                            "by": self.by,
                            "by_name": self._by_name,
                            "by_is_constant": bool(self._by_is_constant),
                            "constraint_mode": self.constraint_mode,
                            "pc": self.pc,
                            "knots": self.knots,
                            "m": self.m,
                            "fixed": bool(self.fixed),
                            "is_selection_penalty": True,
                        },
                    )
                )

        return defs

    def transform_new(self, X_new):
        if self._feature_index is None or self._ps_knots is None:
            raise RuntimeError("Term is not fitted.")

        X_new = np.asarray(X_new, dtype=np.float64)
        xj = X_new[:, self._feature_index]

        B = pspline_predict_matrix(
            xj,
            self._ps_knots,
            basis_order=self.m[0],
            deriv=0,
        )

        if self._by_index is not None:
            z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
            B = B * z[:, None]

        return B
