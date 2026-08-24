"""Gaussian-process smooth term (``bs='gp'``)."""

from __future__ import annotations

import numpy as np

from ...constraints.absorption import (
    fit_single_penalty_with_constraint_policy,
    fit_single_penalty_with_setup_basis,
)
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.univariate.gp import (
    build_gaussian_process_setup,
    predict_gaussian_process,
)
from ..registry import register_smooth
from ..smooth_base import BaseSmoothTerm, _resolve_feature, columns_as_float_matrix


@register_smooth("gp")
class GaussianProcessTerm(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "gp"
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
        xt=None,
        null_penalty_tol=1e-10,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        super().__init__(
            feature=features,
            label=label or f"s({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = int(k)
        self.m = m
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.xt = xt
        self.null_penalty_tol = float(null_penalty_tol)

        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_indices = None
        self._feature_names = None
        self._by_state = None
        self._basis_train = None
        self._penalties = None
        self._setup = None

    @property
    def expected_linked_penalty_count(self):
        return None if self.select else 1

    def fit(self, X, feature_names):
        feature_indices = []
        resolved_names = []
        for feature in self.feature:
            index, name = _resolve_feature(feature, feature_names)
            feature_indices.append(index)
            resolved_names.append(name)

        values = columns_as_float_matrix(X, feature_indices)
        self._set_by_state(X, feature_names)
        self._feature_indices = feature_indices
        self._feature_names = resolved_names
        self._set_resolved_features(resolved_names)

        shared_X = self._linked_id_setup_matrix(feature_names)
        if shared_X is not None:
            setup_values = columns_as_float_matrix(shared_X, feature_indices)
        else:
            setup_values = values
        self._setup = build_gaussian_process_setup(
            setup_values,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        base = (
            setup_base
            if shared_X is None
            else np.asarray(
                predict_gaussian_process(values, self._setup), dtype=np.float64
            )
        )
        raw_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        penalty = scale_penalty(setup_base, raw_penalty)
        self._set_penalty_rescale_factors(
            [penalty_rescale_factor(setup_base, raw_penalty)]
        )

        if self.pc is not None:
            constrained_basis, constrained_penalties, transform, _ = (
                self._apply_point_constraint(
                    base,
                    [penalty],
                    self.pc,
                    feature_names=self._feature_names,
                    point_basis_fn=lambda points: predict_gaussian_process(
                        points, self._setup
                    )[0],
                    fixed=self.fixed,
                )
            )
            self._basis_train = np.asarray(constrained_basis, dtype=np.float64)
            self._penalties = constrained_penalties
            self._record_constraint_result("pc", transform, absorbed_by="runtime")
            return self

        auto_constrain = bool(self._by_state.is_constant)
        if shared_X is None:
            result = fit_single_penalty_with_constraint_policy(
                base,
                penalty,
                self._by_state,
                constraint_mode=self.constraint_mode,
                fixed=self.fixed,
                auto_constrain_when=auto_constrain,
            )
        else:
            result = fit_single_penalty_with_setup_basis(
                base,
                setup_base,
                penalty,
                self._by_state,
                constraint_mode=self.constraint_mode,
                fixed=self.fixed,
                auto_constrain_when=auto_constrain,
            )
        self._basis_train = result.basis_train
        self._penalties = result.penalties
        self._record_constraint_result(
            result.constraint_kind,
            result.constraint_transform,
            absorbed_by=(
                "runtime" if result.constraint_transform is not None else None
            ),
        )
        return self

    def get_penalty_definitions(self):
        self._require_fitted()
        if not self.penalties:
            return []
        metadata = {
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "feature": list(self.feature),
            "label": self.label,
            "by": self.by,
            "by_name": self._by_state.feature_name,
            "by_is_constant": bool(self._by_state.is_constant),
            "constraint_mode": self.constraint_mode,
            "constraint_kind": self.constraint_kind,
            "pc": self.pc,
            "knots": self.knots,
            "xt": self.xt,
            "m": self.m,
            "gp_definition": np.asarray(self._setup.definition).tolist(),
            "original_null_space_dim": self._setup.null_space_dim,
            "fixed": bool(self.fixed),
        }
        selection_metadata = {**metadata, "is_selection_penalty": True}
        metadata = self._penalty_metadata_with_scale(metadata, penalty_index=0)
        return self._build_penalty_block(
            self.penalties[0],
            rank=int(self._setup.rank),
            smooth_metadata=metadata,
            selection_metadata=selection_metadata,
        )

    def transform_new(self, X_new):
        self._require_fitted()
        values = columns_as_float_matrix(X_new, self._feature_indices)
        basis = predict_gaussian_process(values, self._setup)
        return self._apply_constraint_transform_and_by(basis, X_new)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        self._require_fitted()
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        setup_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        if centered:
            if (
                self._linked_id_setup() is not None
                and self.constraint_transform is not None
            ):
                transform = np.asarray(self.constraint_transform, dtype=np.float64)
                scaled = scale_penalty(setup_base, setup_penalty)
                return (
                    np.asarray(setup_base @ transform, dtype=np.float64),
                    np.asarray(transform.T @ scaled @ transform, dtype=np.float64),
                    None,
                )
            return super().tensor_marginal_fit_matrices(centered=True)
        return setup_base, setup_penalty, None

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        if centered:
            basis = np.asarray(self.transform_new(X_new), dtype=np.float64)
        else:
            values = columns_as_float_matrix(X_new, self._feature_indices)
            basis = predict_gaussian_process(values, self._setup)
        if np_transform is not None:
            basis = basis @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(basis, dtype=np.float64)


__all__ = ["GaussianProcessTerm"]
