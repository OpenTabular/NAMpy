"""Derivative-penalized B-spline smooth term (``bs='bs'``)."""

from __future__ import annotations

import numpy as np

from ...constraints.absorption import (
    apply_linear_constraint,
    should_apply_identifiability_constraint,
)
from ...penalties import (
    PenaltySpec,
    normalize_penalty_spec,
    penalty_id_for_local_index,
)
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.univariate.bs import (
    build_derivative_bspline_setup,
    normalize_bspline_orders,
    predict_derivative_bspline,
)
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    build_penalty_definition,
    by_values_from_new_data,
    column_as_numeric_array,
    linear_functional_basis,
    linear_functional_by_state,
)


@register_smooth("bs")
class DerivativeBSplineTerm1D(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "bs"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        m=None,
        label=None,
        term_id=None,
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
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = int(k)
        self.m = normalize_bspline_orders(m)
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.null_penalty_tol = float(null_penalty_tol)

        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_index = None
        self._feature_name = None
        self._by_state = None
        self._basis_train = None
        self._penalties = None
        self._setup = None
        self._linear_functional = False

    @property
    def n_main_penalties(self):
        return 0 if self.fixed else len(self.m) - 1

    @property
    def expected_linked_penalty_count(self):
        return None if self.select else self.n_main_penalties

    def _fit_constraint_policy(self, base, setup_base, penalties):
        penalties_in = [] if self.fixed else list(penalties)
        mode = self.constraint_mode
        if mode == "factor_by" and not self._by_state.is_present:
            raise ValueError(
                "constraint_mode='factor_by' requires a numeric indicator `by` column."
            )
        should_constrain = (
            mode == "factor_by"
            or should_apply_identifiability_constraint(
                self._by_state,
                mode,
                default_when_auto=True,
            )
        )
        if should_constrain:
            _, transformed, transform = apply_linear_constraint(
                setup_base,
                penalties_in,
                np.asarray(setup_base, dtype=np.float64).mean(axis=0),
            )
            base_out = np.asarray(base, dtype=np.float64) @ transform
            kind = "factor_by" if mode == "factor_by" else "sum_to_zero"
        else:
            transformed = penalties_in
            transform = None
            base_out = np.asarray(base, dtype=np.float64)
            kind = None
        base_out = self._apply_cached_by(base_out)
        self._basis_train = np.asarray(base_out, dtype=np.float64)
        self._penalties = [np.asarray(S, dtype=np.float64) for S in transformed]
        self._record_constraint_result(
            kind,
            transform,
            absorbed_by=("runtime" if transform is not None else None),
        )

    def fit(self, X, feature_names):
        self._X_train = np.asarray(X, dtype=object).copy()
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._set_resolved_features([feature_name])
        x_values = column_as_numeric_array(X, idx)
        self._set_by_state(X, feature_names)

        self._linear_functional = np.asarray(x_values).ndim == 2
        if self._linear_functional:
            if self._by_state is None or np.asarray(self._by_state.values).ndim != 2:
                raise ValueError(
                    "B-spline linear-functional terms require matrix-valued by weights."
                )
            by_weights = np.asarray(self._by_state.values, dtype=np.float64)
            if by_weights.shape != np.asarray(x_values).shape:
                raise ValueError(
                    "Linear-functional feature locations and by weights must have equal shape."
                )
            setup_values = np.asarray(x_values, dtype=np.float64).reshape(-1)
            self._by_state = linear_functional_by_state(self._by_state)
        else:
            setup_values = np.asarray(x_values, dtype=np.float64).reshape(-1)

        shared_X = self._linked_id_setup_matrix(feature_names)
        pooled_setup = shared_X is not None
        if pooled_setup:
            if self._linear_functional:
                raise NotImplementedError(
                    "Linked-id pooling is not available for linear-functional B-splines."
                )
            setup_values = np.asarray(
                column_as_numeric_array(shared_X, idx), dtype=np.float64
            ).reshape(-1)

        self._setup = build_derivative_bspline_setup(
            setup_values,
            feature_index=idx,
            feature_name=feature_name,
            bs_dim=self.k,
            m=self.m,
            knots=self.knots,
        )
        point_base = np.asarray(
            predict_derivative_bspline(x_values, self._setup), dtype=np.float64
        )
        if self._linear_functional:
            base = linear_functional_basis(
                x_values,
                by_weights,
                lambda values: predict_derivative_bspline(values, self._setup),
            )
            setup_base = np.asarray(base, dtype=np.float64)
        else:
            base = point_base
            setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)

        raw_penalties = [np.asarray(S, dtype=np.float64) for S in self._setup.penalties]
        scales = [penalty_rescale_factor(setup_base, S) for S in raw_penalties]
        self._set_penalty_rescale_factors(scales)
        scaled_penalties = [scale_penalty(setup_base, S) for S in raw_penalties]

        if self.pc is not None:
            Bc, Sc, C, _ = self._apply_point_constraint(
                base,
                scaled_penalties,
                self.pc,
                feature_names=[self._feature_name],
                point_basis_fn=lambda pts: predict_derivative_bspline(pts, self._setup)[
                    0
                ],
                fixed=self.fixed,
            )
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        self._fit_constraint_policy(base, setup_base, scaled_penalties)
        return self

    @property
    def basis_train(self):
        self._require_fitted()
        return self._basis_train

    @property
    def penalties(self):
        self._require_fitted()
        return self._penalties

    @property
    def n_coef(self):
        self._require_fitted()
        return int(self._basis_train.shape[1])

    def get_penalty_definitions(self):
        self._require_fitted()
        raw = list(self.penalties)
        if not raw:
            return []
        selection_defs = self._build_selection_penalty_definitions(
            raw,
            null_penalty_tol=self.null_penalty_tol,
        )
        sp_vals = self._normalized_term_sp(len(raw) + len(selection_defs))
        definitions = []
        for j, penalty in enumerate(raw):
            sid = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(
                    self.smoothing_id, j, n_penalties=len(raw)
                )
            )
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            definitions.append(
                build_penalty_definition(
                    self,
                    penalty,
                    kind="smooth",
                    smoothing_id=sid,
                    sp_value_in=sp_j,
                    metadata_extra={
                        "term_sp": sp_j,
                        "m": self.m,
                        "derivative_order": self.m[j + 1],
                        "is_selection_penalty": False,
                    },
                    local_penalty_index=j,
                )
            )

        for offset, selection in enumerate(selection_defs, start=len(raw)):
            sp_j = sp_vals[offset] if offset < len(sp_vals) else None
            if sp_j is None:
                definitions.append(selection)
                continue
            definitions.append(
                normalize_penalty_spec(
                    PenaltySpec(
                        matrix=np.asarray(selection.matrix, dtype=np.float64),
                        smoothing_id=selection.smoothing_id,
                        kind=selection.kind,
                        rank=selection.rank,
                        null_space_dim=selection.null_space_dim,
                        is_null_space_penalty=selection.is_null_space_penalty,
                        sp_mode="fixed" if sp_j >= 0 else "estimate",
                        sp_value=float(sp_j) if sp_j >= 0 else None,
                        metadata=dict(selection.metadata),
                    )
                )
            )
        return definitions

    def transform_new(self, X_new):
        self._require_fitted()
        x_values = column_as_numeric_array(X_new, self._feature_index)
        if self._linear_functional:
            basis = linear_functional_basis(
                x_values,
                by_values_from_new_data(X_new, self._by_state),
                lambda values: predict_derivative_bspline(values, self._setup),
            )
            if self.constraint_transform is not None:
                basis = basis @ self.constraint_transform
            return np.asarray(basis, dtype=np.float64)
        basis = predict_derivative_bspline(x_values, self._setup)
        return self._apply_constraint_transform_and_by(basis, X_new)

    def derivative_matrix(self, X_new=None, order=1):
        self._require_fitted()
        order = int(order)
        if order < 1 or order > int(self._setup.degree):
            raise ValueError(f"order must be between 1 and {self._setup.degree}.")
        if self._linear_functional:
            raise NotImplementedError(
                "Derivatives of linear-functional B-spline terms require an "
                "explicit functional derivative and are not inferred."
            )
        source = self._X_train if X_new is None else X_new
        x_values = column_as_numeric_array(source, self._feature_index)
        basis = predict_derivative_bspline(x_values, self._setup, deriv=order)
        return self._apply_constraint_transform_and_by(basis, source)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        if len(self._setup.penalties) != 1:
            raise NotImplementedError(
                "Sorry, tensor products of smooths with multiple penalties are not "
                "supported."
            )
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        raw_penalty = np.asarray(self._setup.penalties[0], dtype=np.float64)
        if centered:
            return super().tensor_marginal_fit_matrices(centered=True)
        return setup_base, raw_penalty, None

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        if centered:
            basis = np.asarray(self.transform_new(X_new), dtype=np.float64)
        else:
            x_values = column_as_numeric_array(X_new, self._feature_index)
            if np.asarray(x_values).ndim != 1:
                raise NotImplementedError(
                    "Matrix-valued B-splines cannot be tensor marginals."
                )
            basis = predict_derivative_bspline(x_values, self._setup)
        if np_transform is not None:
            basis = basis @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(basis, dtype=np.float64)
