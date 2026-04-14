"""
Cubic regression spline smooth term (``bs='cr'``).

Implements the :class:`BaseSmoothTerm` interface for a single-variable cubic
regression spline basis with a second-derivative penalty.  Knots are either
user-supplied or chosen automatically using quantiles of the training data.

An optional null-space selection penalty is added when ``select=True``,
allowing the term to be shrunk to zero.

A point constraint ``coef[i] = 0`` may be specified via the ``pc`` argument
to pin a basis function to a fixed value (used for derivative computation and
identifiability in some settings).
"""

import numpy as np

from ....splines.cubic import CubicSplines
from ....splines.univariate_bases import (
    add_full_rank_shrinkage,
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
    place_knots_through_values,
)
from ...constraints.absorption import apply_linear_constraint
from ...penalties.algebra import scale_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_point_constraint,
    _resolve_feature,
    column_as_float,
    resolve_by_state,
    sync_by_state_attributes,
)


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
        term_id=None,
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
            term_id=term_id,
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

        self._by_state = None

        self._basis_train = None
        self._penalties = None
        self._use_centered_basis = True

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

        if self.basis_name == "cs":
            # Mirror mgcv smooth.construct.cr.smooth.spec: shrinkage is applied to
            # the *unscaled* raw penalty, then the result is scaled and (for the
            # centered path) projected through the centering matrix.  Applying
            # shrinkage after scaling changes the normalisation denominator and
            # produces a ~1e-5 prediction error.
            S_unscaled = np.asarray(self._spline.raw_penalty_unscaled, dtype=np.float64)
            S = add_full_rank_shrinkage(S_unscaled, shrink=0.1)
            S = scale_penalty(self._spline.raw_basis, S)
            if not raw:
                C = self._spline.center_mat
                S = C.T @ S @ C
            return 0.5 * (S + S.T)

        S = self._spline.raw_penalty if raw else self._spline.penalty
        S = np.asarray(S, dtype=np.float64)
        return 0.5 * (S + S.T)

    def fit(self, X, feature_names):
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._set_resolved_features([feature_name])

        xj = column_as_float(X, idx)

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        self._pc_value = None

        if self.basis_name in {"cr", "cs"}:
            self._pooled_linked_raw_marginal = False
            if self.shared_basis_setup is not None:
                mode = self.shared_basis_setup.get("mode", None)
                if mode != "pooled_cr_1d":
                    raise NotImplementedError(
                        f"Unknown shared basis setup mode {mode!r}."
                    )
                pooled_knots = np.asarray(
                    self.shared_basis_setup.get("pooled_knots", []),
                    dtype=np.float64,
                ).ravel()
                if pooled_knots.size == 0:
                    raise ValueError(
                        "shared_basis_setup for pooled_cr_1d must contain non-empty pooled_knots."
                    )
                self._spline = CubicSplines(xj, self.k, knots=pooled_knots)
                # Restore pooled-data penalty matrices so smoothing parameter
                # optimisation uses the same penalty scale as when the shared
                # basis was built from all linked terms' data combined.
                pooled_raw_pen = self.shared_basis_setup.get("pooled_raw_penalty")
                pooled_center = self.shared_basis_setup.get("pooled_center_mat")
                pooled_pen = self.shared_basis_setup.get("pooled_penalty")
                if pooled_raw_pen is not None:
                    self._spline.raw_penalty = np.asarray(
                        pooled_raw_pen, dtype=np.float64
                    )
                if pooled_center is not None:
                    self._spline.center_mat = np.asarray(
                        pooled_center, dtype=np.float64
                    )
                if pooled_pen is not None:
                    self._spline.penalty = np.asarray(pooled_pen, dtype=np.float64)
                pooled_setup = True
            else:
                self._spline = CubicSplines(xj, self.k, knots=self.knots)
                pooled_setup = False

            if self.pc is not None:
                self._pc_value = _normalize_point_constraint(
                    self.pc, self._feature_name
                )
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
                Bc, Sc, C = apply_linear_constraint(
                    raw_base,
                    penalties_in,
                    pc_basis,
                )

                if self._by_state.is_present:
                    Bc = Bc * self._by_state.values[:, None]

                self._basis_train = np.asarray(Bc, dtype=np.float64)
                self._penalties = Sc
                self._record_constraint_result("pc", C, absorbed_by="runtime")
                self._use_centered_basis = False
                return self

            if self.constraint_mode == "factor_by":
                if not self._by_state.is_present:
                    raise ValueError(
                        "constraint_mode='factor_by' requires a numeric indicator `by` column."
                    )

                raw_base = (
                    self._spline.transform_new_raw(xj)
                    if pooled_setup
                    else self._spline.raw_basis
                )
                main_penalty = self._main_penalty(raw=True)

                # mgcv centers the raw basis first (shared sum-to-zero over all
                # observations), then scales by the level indicator.  Applying the
                # constraint to the indicator-scaled basis produces a different
                # constraint direction and breaks parity.
                penalties_in = [] if self.fixed else [main_penalty]
                mean_row = raw_base.mean(axis=0)
                Bc_raw, Sc, C = apply_linear_constraint(
                    raw_base,
                    penalties_in,
                    mean_row,
                )
                base = Bc_raw * self._by_state.values[:, None]

                self._basis_train = np.asarray(base, dtype=np.float64)
                self._penalties = Sc
                self._record_constraint_result("factor_by", C, absorbed_by="runtime")
                self._use_centered_basis = False
                return self

            if self.constraint_mode == "always":
                self._use_centered_basis = True
            elif self.constraint_mode == "never":
                self._use_centered_basis = False
            else:
                self._use_centered_basis = bool(self._by_state.is_constant)

            if self._use_centered_basis:
                if pooled_setup:
                    # Knots / identconst for linked `id=` smooths are built from pooled
                    # covariates; centered columns from transform_new_centered(xj) need
                    # not sum to zero on each term's row subset, so stage 5 would apply a
                    # second sum-to-zero and drop an extra column vs mgcv. Use raw subset
                    # basis and let apply_global_side_conditions perform the single
                    # centering (same pattern as the non-centered branch).
                    base = self._spline.transform_new_raw(xj)
                    pen = self._main_penalty(raw=True)
                    self._pooled_linked_raw_marginal = True
                else:
                    base = self._spline.basis
                    pen = self._main_penalty(raw=False)
            else:
                base = (
                    self._spline.transform_new_raw(xj)
                    if pooled_setup
                    else self._spline.raw_basis
                )
                pen = self._main_penalty(raw=True)

            if self._by_state.is_present:
                base = base * self._by_state.values[:, None]

            self._basis_train = np.asarray(base, dtype=np.float64)
            self._penalties = [] if self.fixed else [np.asarray(pen, dtype=np.float64)]
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        if self.shared_basis_setup is not None:
            raise NotImplementedError(
                "shared_basis_setup is not yet implemented for bs='cc'."
            )
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

        self._cc_knots = np.asarray(k, dtype=np.float64)
        self._cc_bd = np.asarray(BD, dtype=np.float64)

        if self.pc is not None:
            pc_value = _normalize_point_constraint(self.pc, self._feature_name)
            pc_basis = cyclic_cubic_predict_matrix(
                np.asarray([pc_value], dtype=np.float64), k, BD
            )[0]
            S_raw = D.T @ BD
            main_penalty = 0.5 * (S_raw + S_raw.T)
            penalties_in = [] if self.fixed else [main_penalty]
            Bc, Sc, C = apply_linear_constraint(base, penalties_in, pc_basis)
            if self._by_state.is_present:
                Bc = Bc * self._by_state.values[:, None]
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._use_centered_basis = False
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        S_raw = D.T @ BD
        S_sym = 0.5 * (S_raw + S_raw.T)

        if self.constraint_mode == "never":
            # Tensor-marginal path: return raw k-column basis without absorbing
            # the sum-to-zero constraint (mgcv smoothCon absorb.cons=FALSE).
            main_penalty = scale_penalty(base, S_sym)
            penalties_in = [] if self.fixed else [main_penalty]
            if self._by_state.is_present:
                base = base * self._by_state.values[:, None]
            self._basis_train = np.asarray(base, dtype=np.float64)
            self._penalties = penalties_in
            self._use_centered_basis = False
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        main_penalty = scale_penalty(base, S_sym)
        penalties_in = [] if self.fixed else [main_penalty]
        mean_row = base.mean(axis=0)
        Bc, Sc, C = apply_linear_constraint(base, penalties_in, mean_row)
        if self._by_state.is_present:
            Bc = Bc * self._by_state.values[:, None]
        self._basis_train = np.asarray(Bc, dtype=np.float64)
        self._penalties = Sc
        self._use_centered_basis = False
        self._record_constraint_result("centering", C, absorbed_by="runtime")
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        if len(self.penalties) == 0:
            return []

        base_metadata = {
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "feature": self.feature,
            "label": self.label,
            "by": self.by,
            "by_name": self._by_state.feature_name,
            "by_is_constant": bool(self._by_state.is_constant),
            "constraint_mode": self.constraint_mode,
            "constraint_kind": self.constraint_kind,
            "pc": self._pc_value,
            "knots": self.knots,
            "has_shared_basis_setup": self.shared_basis_setup is not None,
            "fixed": bool(self.fixed),
        }
        return self._build_penalty_block(
            self.penalties[0],
            smooth_metadata=base_metadata,
            selection_metadata={**base_metadata, "is_selection_penalty": True},
            selection_via_subsystem=True,
        )

    def transform_new(self, X_new):
        self._require_fitted()

        xj = column_as_float(X_new, self._feature_index)

        if self.basis_name == "cc":
            B = cyclic_cubic_predict_matrix(xj, self._cc_knots, self._cc_bd)
            return self._apply_constraint_transform_and_by(B, X_new)

        if self._use_centered_basis and not getattr(
            self, "_pooled_linked_raw_marginal", False
        ):
            B = self._spline.transform_new_centered(xj)
        else:
            B = self._spline.transform_new_raw(xj)

        return self._apply_constraint_transform_and_by(B, X_new)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        basis_name = str(self.basis_name).lower()
        if basis_name in {"cr", "cs"} and self._spline is not None:
            if centered:
                B = np.asarray(self.basis_train, dtype=np.float64)
                S = np.asarray(self.penalties[0], dtype=np.float64)
                XP = self._spline._np_transform_centered
            else:
                B = np.asarray(self._spline.raw_basis, dtype=np.float64)
                S = np.asarray(self._spline.raw_penalty_unscaled, dtype=np.float64)
                XP = self._spline._np_transform
            if XP is not None:
                B = B @ XP
                S = XP.T @ S @ XP
            return B, S, XP
        return super().tensor_marginal_fit_matrices(centered=centered)

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        basis_name = str(self.basis_name).lower()
        if basis_name in {"cr", "cs"} and self._spline is not None:
            xj = column_as_float(X_new, self._feature_index)
            if centered:
                B = self._spline.transform_new_centered(xj)
                XP = self._spline._np_transform_centered
            else:
                B = self._spline.transform_new_raw(xj)
                XP = self._spline._np_transform
            if XP is not None:
                B = B @ XP
        else:
            B = np.asarray(self.transform_new(X_new), dtype=np.float64)
        if np_transform is not None:
            B = B @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(B, dtype=np.float64)
