import numpy as np

from ..base import (
    BaseSmoothTerm,
    _resolve_feature,
    _resolve_numeric_by,
    _is_effectively_constant,
    _apply_linear_constraint,
)
from ..registry import register_smooth
from ...design.objects import PenaltyDefinition
from ...design.penalties import null_space_penalty_from_penalty
from ....splines.thin_plate import build_tprs_term_setup, predict_tprs_term


@register_smooth("tp")
@register_smooth("ts")
class ThinPlateSplineTerm(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "tp"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        basis="tp",
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
        xt=None,
        null_penalty_tol=1e-10,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]

        super().__init__(
            feature=features,
            label=label or f"s({', '.join(map(str, features))})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.k = int(k)
        self.basis_name = str(basis).lower()
        self.m = m
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.xt = xt
        self.null_penalty_tol = float(null_penalty_tol)

        if self.basis_name not in {"tp", "ts"}:
            raise NotImplementedError(
                f"ThinPlateSplineTerm currently supports only basis in "
                f"{{'tp','ts'}}, got {basis!r}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_indices = None
        self._feature_names = None

        self._by_index = None
        self._by_name = None
        self._by_values = None
        self._by_is_constant = None

        self._basis_train = None
        self._penalties = None

        self._constraint_kind = None
        self._constraint_transform = None
        self._setup = None

    def fit(self, X, feature_names):
        feature_indices = []
        feature_names_resolved = []

        for feat in self.feature:
            idx, fname = _resolve_feature(feat, feature_names)
            feature_indices.append(idx)
            feature_names_resolved.append(fname)

        X = np.asarray(X)
        Xf = np.column_stack([np.asarray(X[:, idx], dtype=np.float64) for idx in feature_indices])

        if self.by is not None:
            self._by_index, self._by_name, self._by_values = _resolve_numeric_by(
                self.by, X, feature_names
            )
            self._by_is_constant = _is_effectively_constant(self._by_values)
        else:
            self._by_index, self._by_name, self._by_values = None, None, None
            self._by_is_constant = True

        if self.pc is not None:
            raise NotImplementedError(
                "pc=... is not yet implemented for bs='tp'/'ts' in this step."
            )

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved

        self._setup = build_tprs_term_setup(
            Xf,
            basis=self.basis_name,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )

        base = np.asarray(self._setup.basis_train, dtype=np.float64)
        pen = np.asarray(self._setup.penalty, dtype=np.float64)

        self._constraint_kind = None
        self._constraint_transform = None

        if self.constraint_mode == "factor_by":
            if self._by_values is None:
                raise ValueError(
                    "constraint_mode='factor_by' requires a numeric indicator `by` column."
                )

            base_fb = base * self._by_values[:, None]
            penalties_in = [] if self.fixed else [pen]
            mean_row = base_fb.mean(axis=0)
            Bc, Sc, C = _apply_linear_constraint(base_fb, penalties_in, mean_row)

            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._constraint_kind = "factor_by"
            self._constraint_transform = C
            return self

        should_constrain = False
        if self.constraint_mode == "always":
            should_constrain = True
        elif self.constraint_mode == "never":
            should_constrain = False
        else:
            should_constrain = (
                self.basis_name == "tp"
                and not self._setup.drop_null_effective
                and bool(self._by_is_constant)
            )

        if should_constrain:
            penalties_in = [] if self.fixed else [pen]
            mean_row = base.mean(axis=0)
            Bc, Sc, C = _apply_linear_constraint(base, penalties_in, mean_row)
            base = Bc
            penalties_out = Sc
            self._constraint_kind = "sum_to_zero"
            self._constraint_transform = C
        else:
            penalties_out = [] if self.fixed else [pen]

        if self._by_values is not None:
            base = base * self._by_values[:, None]

        self._basis_train = np.asarray(base, dtype=np.float64)
        self._penalties = penalties_out
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
                rank=int(self._setup.rank) if self._setup is not None else None,
                sp_mode=sp_mode,
                sp_value=sp_value,
                metadata={
                    "term_type": self.term_type,
                    "basis_name": self.basis_name,
                    "feature": list(self.feature),
                    "label": self.label,
                    "by": self.by,
                    "by_name": self._by_name,
                    "by_is_constant": bool(self._by_is_constant),
                    "constraint_mode": self.constraint_mode,
                    "constraint_kind": self._constraint_kind,
                    "pc": self.pc,
                    "knots": self.knots,
                    "xt": self.xt,
                    "m": self.m,
                    "penalty_order": None if self._setup is None else self._setup.penalty_order,
                    "original_null_space_dim": None if self._setup is None else self._setup.original_null_space_dim,
                    "drop_null_requested": None if self._setup is None else bool(self._setup.drop_null_requested),
                    "drop_null_effective": None if self._setup is None else bool(self._setup.drop_null_effective),
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
                            "feature": list(self.feature),
                            "label": self.label,
                            "by": self.by,
                            "by_name": self._by_name,
                            "by_is_constant": bool(self._by_is_constant),
                            "constraint_mode": self.constraint_mode,
                            "constraint_kind": self._constraint_kind,
                            "pc": self.pc,
                            "knots": self.knots,
                            "xt": self.xt,
                            "m": self.m,
                            "penalty_order": None if self._setup is None else self._setup.penalty_order,
                            "original_null_space_dim": None if self._setup is None else self._setup.original_null_space_dim,
                            "drop_null_requested": None if self._setup is None else bool(self._setup.drop_null_requested),
                            "drop_null_effective": None if self._setup is None else bool(self._setup.drop_null_effective),
                            "fixed": bool(self.fixed),
                            "is_selection_penalty": True,
                        },
                    )
                )

        return defs

    def transform_new(self, X_new):
        if self._feature_indices is None or self._setup is None:
            raise RuntimeError("Term is not fitted.")

        X_new = np.asarray(X_new)
        Xf_new = np.column_stack(
            [np.asarray(X_new[:, idx], dtype=np.float64) for idx in self._feature_indices]
        )

        B = predict_tprs_term(Xf_new, self._setup)

        if self._constraint_kind == "factor_by":
            if self._by_index is not None:
                z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
                B = B * z[:, None]
            if self._constraint_transform is not None:
                B = B @ self._constraint_transform
            return B

        if self._constraint_transform is not None:
            B = B @ self._constraint_transform

        if self._by_index is not None:
            z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
            B = B * z[:, None]

        return B
