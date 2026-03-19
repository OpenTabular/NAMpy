import numpy as np

from ..base import (
    BaseSmoothTerm,
    _resolve_feature,
    _resolve_numeric_by,
    _is_effectively_constant,
    _normalize_point_constraint,
    _apply_linear_constraint,
)
from ..registry import register_smooth
from ...design.objects import PenaltyDefinition
from ...design.penalties import null_space_penalty_from_penalty
from ....splines.cubic import CubicSplines
from ....splines.univariate_bases import (
    add_full_rank_shrinkage,
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
    place_knots_through_values,
)


@register_smooth("linear")
class LinearTerm(BaseSmoothTerm):
    term_type = "parametric"
    basis_name = "linear"
    supports_tensor_marginal = False

    def __init__(self, feature, label=None, metadata=None):
        super().__init__(
            feature=feature,
            label=label,
            smoothing_id=None,
            by=None,
            metadata=metadata,
        )
        self._feature_index = None
        self._feature_name = None
        self._basis_train = None

    def fit(self, X, feature_names):
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._basis_train = np.asarray(X[:, idx : idx + 1], dtype=np.float64)
        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        return []

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return 1

    def transform_new(self, X_new):
        if self._feature_index is None:
            raise RuntimeError("Term is not fitted.")
        X_new = np.asarray(X_new, dtype=np.float64)
        return X_new[:, self._feature_index : self._feature_index + 1]


@register_smooth("spline_1d")
@register_smooth("cr")
@register_smooth("cs")
@register_smooth("cc")
class SplineTerm1D(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "cr"
    supports_tensor_marginal = True

    def __init__(
        self,
        feature,
        k=10,
        basis="cr",
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        constraint_mode="auto",
        shared_basis_setup=None,
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
        self.shared_basis_setup = shared_basis_setup
        self.pc = pc
        self.knots = knots
        self.null_penalty_tol = float(null_penalty_tol)

        self._feature_index = None
        self._feature_name = None
        self._spline = None

        self._by_index = None
        self._by_name = None
        self._by_values = None
        self._by_is_constant = None

        self._basis_train = None
        self._penalties = None
        self._use_centered_basis = True

        self._constraint_kind = None
        self._constraint_transform = None
        self._pc_value = None

        self._cc_knots = None
        self._cc_bd = None

        if self.k < 3 and self.basis_name in {"cr", "cs"}:
            raise ValueError("k must be >= 3 for cubic regression spline terms.")
        if self.k < 4 and self.basis_name == "cc":
            raise ValueError("k must be >= 4 for cyclic cubic spline terms.")
        if self.basis_name not in {"cr", "cs", "cc"}:
            raise NotImplementedError(
                f"SplineTerm1D currently supports only basis in "
                f"{{'cr','cs','cc'}}, got {basis!r}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

    def _main_penalty(self, *, raw: bool):
        if self.basis_name == "cc":
            if self._cc_knots is None:
                raise RuntimeError("Term is not fitted.")
            _, _, D = cyclic_cubic_bd(self._cc_knots)
            BD, _, _ = cyclic_cubic_bd(self._cc_knots)
            S = D.T @ BD
            return 0.5 * (S + S.T)

        if self._spline is None:
            raise RuntimeError("Term is not fitted.")

        S = self._spline.raw_penalty if raw else self._spline.penalty
        S = np.asarray(S, dtype=np.float64)

        if self.basis_name == "cs":
            S = add_full_rank_shrinkage(S, shrink=0.1)

        return 0.5 * (S + S.T)

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

        self._constraint_kind = None
        self._constraint_transform = None
        self._pc_value = None

        if self.basis_name in {"cr", "cs"}:
            if self.shared_basis_setup is not None:
                mode = self.shared_basis_setup.get("mode", None)
                if mode != "pooled_cr_1d":
                    raise NotImplementedError(
                        f"Unknown shared basis setup mode {mode!r}."
                    )
                setup_x = np.asarray(
                    self.shared_basis_setup.get("pooled_x", []),
                    dtype=np.float64,
                ).ravel()
                if setup_x.size == 0:
                    raise ValueError(
                        "shared_basis_setup for pooled_cr_1d must contain non-empty pooled_x."
                    )
                self._spline = CubicSplines(setup_x, self.k, knots=self.knots)
                pooled_setup = True
            else:
                self._spline = CubicSplines(xj, self.k, knots=self.knots)
                pooled_setup = False

            if self.pc is not None:
                if self.constraint_mode == "factor_by":
                    raise NotImplementedError(
                        "pc=... is not yet implemented for factor-by replicated smooths."
                    )

                self._pc_value = _normalize_point_constraint(self.pc, self._feature_name)
                raw_base = (
                    self._spline.transform_new_raw(xj)
                    if pooled_setup
                    else self._spline.raw_basis
                )
                main_penalty = self._main_penalty(raw=True)

                pc_basis = self._spline.transform_new_raw(
                    np.asarray([self._pc_value], dtype=np.float64)
                )[0]

                penalties_in = [] if self.fixed else [main_penalty]
                Bc, Sc, C = _apply_linear_constraint(
                    raw_base,
                    penalties_in,
                    pc_basis,
                )

                if self._by_values is not None:
                    Bc = Bc * self._by_values[:, None]

                self._basis_train = np.asarray(Bc, dtype=np.float64)
                self._penalties = Sc
                self._constraint_kind = "pc"
                self._constraint_transform = C
                self._use_centered_basis = False
                return self

            if self.constraint_mode == "factor_by":
                if self._by_values is None:
                    raise ValueError(
                        "constraint_mode='factor_by' requires a numeric indicator `by` column."
                    )

                raw_base = (
                    self._spline.transform_new_raw(xj)
                    if pooled_setup
                    else self._spline.raw_basis
                )
                base = raw_base * self._by_values[:, None]
                main_penalty = self._main_penalty(raw=True)

                penalties_in = [] if self.fixed else [main_penalty]
                mean_row = base.mean(axis=0)
                Bc, Sc, C = _apply_linear_constraint(
                    base,
                    penalties_in,
                    mean_row,
                )

                self._basis_train = np.asarray(Bc, dtype=np.float64)
                self._penalties = Sc
                self._constraint_kind = "factor_by"
                self._constraint_transform = C
                self._use_centered_basis = False
                return self

            if self.constraint_mode == "always":
                self._use_centered_basis = True
            elif self.constraint_mode == "never":
                self._use_centered_basis = False
            else:
                self._use_centered_basis = bool(self._by_is_constant)

            if self._use_centered_basis:
                base = (
                    self._spline.transform_new_centered(xj)
                    if pooled_setup
                    else self._spline.basis
                )
                pen = self._main_penalty(raw=False)
            else:
                base = (
                    self._spline.transform_new_raw(xj)
                    if pooled_setup
                    else self._spline.raw_basis
                )
                pen = self._main_penalty(raw=True)

            if self._by_values is not None:
                base = base * self._by_values[:, None]

            self._basis_train = np.asarray(base, dtype=np.float64)
            self._penalties = [] if self.fixed else [np.asarray(pen, dtype=np.float64)]
            return self

        if self.shared_basis_setup is not None:
            raise NotImplementedError(
                "shared_basis_setup is not yet implemented for bs='cc'."
            )
        if self.pc is not None:
            raise NotImplementedError("pc=... is not yet implemented for bs='cc'.")
        if self.constraint_mode == "factor_by":
            raise NotImplementedError(
                "factor-by replicated cyclic cubic smooths are not yet implemented."
            )

        k = self.knots
        if k is None:
            k = place_knots_through_values(xj, self.k)
        else:
            k = np.asarray(k, dtype=np.float64).ravel()
            if k.size == 2:
                k = place_knots_through_values(np.concatenate([k, xj]), self.k)
            elif k.size != self.k:
                raise ValueError("number of supplied knots != k for a cc smooth")

        BD, _, D = cyclic_cubic_bd(k)
        base = cyclic_cubic_predict_matrix(xj, k, BD)

        if self._by_values is not None:
            base = base * self._by_values[:, None]

        self._cc_knots = np.asarray(k, dtype=np.float64)
        self._cc_bd = np.asarray(BD, dtype=np.float64)
        self._basis_train = np.asarray(base, dtype=np.float64)

        if self.fixed:
            self._penalties = []
        else:
            S = D.T @ BD
            self._penalties = [0.5 * (S + S.T)]

        self._use_centered_basis = False
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
                    "constraint_kind": self._constraint_kind,
                    "pc": self._pc_value,
                    "knots": self.knots,
                    "has_shared_basis_setup": self.shared_basis_setup is not None,
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
                            "constraint_kind": self._constraint_kind,
                            "pc": self._pc_value,
                            "knots": self.knots,
                            "has_shared_basis_setup": self.shared_basis_setup is not None,
                            "fixed": bool(self.fixed),
                            "is_selection_penalty": True,
                        },
                    )
                )

        return defs

    def transform_new(self, X_new):
        if self._feature_index is None:
            raise RuntimeError("Term is not fitted.")

        X_new = np.asarray(X_new, dtype=np.float64)
        xj = X_new[:, self._feature_index]

        if self.basis_name == "cc":
            if self._cc_knots is None or self._cc_bd is None:
                raise RuntimeError("Term is not fitted.")
            B = cyclic_cubic_predict_matrix(xj, self._cc_knots, self._cc_bd)
            if self._by_index is not None:
                z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
                B = B * z[:, None]
            return B

        if self._spline is None:
            raise RuntimeError("Term is not fitted.")

        if self._constraint_kind == "factor_by":
            B = self._spline.transform_new_raw(xj)
            if self._by_index is not None:
                z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
                B = B * z[:, None]
            if self._constraint_transform is not None:
                B = B @ self._constraint_transform
            return B

        if self._constraint_kind == "pc":
            B = self._spline.transform_new_raw(xj)
            if self._constraint_transform is not None:
                B = B @ self._constraint_transform
            if self._by_index is not None:
                z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
                B = B * z[:, None]
            return B

        if self._use_centered_basis:
            B = self._spline.transform_new_centered(xj)
        else:
            B = self._spline.transform_new_raw(xj)

        if self._by_index is not None:
            z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
            B = B * z[:, None]

        return B
