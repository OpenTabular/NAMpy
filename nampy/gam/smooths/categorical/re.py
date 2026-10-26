"""
Random-effect smooth term (``bs='re'``).

Implements the :class:`BaseSmoothTerm` interface for a random-effect / variance
component term.  The design matrix is a dummy-coded indicator matrix for the
grouping factor, and the penalty is a scaled identity matrix (i.i.d. random
effects prior).

The smoothing parameter controls the random-effects variance: large lambda
corresponds to a tight prior (small variance, strong shrinkage toward zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...linalg import symmetrize_matrix
from ...penalties import PenaltySpec
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ..algebra import rowwise_kronecker
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    by_values_from_new_data,
    linear_functional_by_state,
)
from .categorical_utils import (
    aggregate_functional_basis,
    all_bool_like,
    as_object_1d,
    factor_indicator_matrix,
    factor_levels_from_metadata,
    functional_object_columns,
    stable_unique_levels,
    try_numeric_1d,
)


@dataclass
class _REComponentSpec:
    kind: str  # "numeric" or "factor"
    levels: list[Any] | None = None


def _fit_re_component(x, *, levels=None):
    """
    Build one component matrix contributing to model.matrix(~x - 1)-style random effects.
    """
    x_obj = as_object_1d(x)

    if any(pd.isna(v) for v in x_obj):
        raise ValueError(
            "Random-effect terms do not currently allow missing values in fitting."
        )

    if levels is not None:
        return factor_indicator_matrix(x_obj, levels), _REComponentSpec(
            kind="factor", levels=list(levels)
        )

    if all_bool_like(x_obj):
        levels = stable_unique_levels(x_obj)
        return factor_indicator_matrix(x_obj, levels), _REComponentSpec(
            kind="factor", levels=levels
        )

    x_num = try_numeric_1d(x_obj)
    if x_num is not None and np.isfinite(x_num).all():
        return x_num[:, None], _REComponentSpec(kind="numeric", levels=None)

    levels = stable_unique_levels(x_obj)
    return factor_indicator_matrix(x_obj, levels), _REComponentSpec(
        kind="factor", levels=levels
    )


def _predict_re_component(x_new, spec: _REComponentSpec):
    x_obj = as_object_1d(x_new)

    if spec.kind == "numeric":
        x_num = try_numeric_1d(x_obj)
        if x_num is None:
            raise ValueError(
                "Expected numeric prediction data for a numeric random-effect component."
            )
        B = np.zeros((x_num.size, 1), dtype=np.float64)
        ok = np.isfinite(x_num)
        B[ok, 0] = x_num[ok]
        return B

    return factor_indicator_matrix(x_obj, spec.levels or [])


def _combine_random_effect_blocks(blocks):
    """
    Mirror `mgcv/R/smooth.r::smooth.construct.re.smooth.spec`, which builds
    `model.matrix(~ term1:term2:... - 1)`.

    For multi-way factor interactions, R's model matrix nests columns with the
    rightmost margin varying slowest. `rowwise_kronecker` varies the leftmost
    margin slowest, so reverse the blocks before forming the interaction design.
    """
    if len(blocks) == 0:
        raise ValueError("blocks must contain at least one matrix.")
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=np.float64)
    return np.asarray(rowwise_kronecker(blocks[::-1]), dtype=np.float64)


@register_smooth("re")
class RandomEffectTerm(BaseSmoothTerm):
    """
    mgcv-style simple random effect smooth:
        s(x, bs="re")
        s(f, bs="re")
        s(x, f, bs="re")
        s(f1, f2, bs="re")
        ...

    Implemented as the parametric interaction design:
        ~ x:f1:f2:... - 1

    with default identity penalty, or user-supplied precision components in xt.

    Important mgcv-compatible behavior
    ----------------------------------
    - random effects do not work with linked smoothing ids via ``id=``
    - no centering constraints
    - unseen factor levels at prediction -> zero rows
    """

    term_type = "random_effect"
    basis_name = "re"
    supports_tensor_marginal = True
    tensor_requires_centering = False
    tensor_no_reparameterization = True

    def __init__(
        self,
        feature,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        xt=None,
        metadata=None,
        k=-1,
        fixed=False,
        constraint_mode="auto",
        knots=None,
    ):
        del k, fixed, constraint_mode, knots
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]

        if smoothing_id is not None:
            raise NotImplementedError("random effects don't work with ids.")

        super().__init__(
            feature=features,
            label=label or f"re({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=None,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.select = bool(select)
        self.xt = xt

        self._feature_indices = None
        self._feature_names = None

        self._by_state = None

        self._component_specs = None
        self._basis_train = None
        self._penalties = None
        self._ranks = None

        # random effects are full-rank penalized: no side conditions needed
        self.skip_centering = True

    def fit(self, X, feature_names):
        X = np.asarray(X, dtype=object)

        feature_indices = []
        feature_names_resolved = []
        blocks = []
        specs = []

        columns = functional_object_columns(
            X, [_resolve_feature(feat, feature_names)[0] for feat in self.feature]
        )
        self._linear_functional = columns[0].ndim == 2
        for feat, column in zip(self.feature, columns, strict=True):
            idx, fname = _resolve_feature(feat, feature_names)
            feature_indices.append(idx)
            feature_names_resolved.append(fname)

            levels = factor_levels_from_metadata(self.metadata, fname)
            B_i, spec_i = _fit_re_component(column.reshape(-1), levels=levels)
            blocks.append(B_i)
            specs.append(spec_i)

        self._set_by_state(X, feature_names)

        B_raw = _combine_random_effect_blocks(blocks)
        if self._linear_functional:
            weights = self._by_state.values
            B = aggregate_functional_basis(B_raw, columns[0].shape, weights)
            self._by_state = linear_functional_by_state(
                self._by_state,
                weights=np.ones(columns[0].shape) if weights is None else weights,
            )
        else:
            B = self._apply_cached_by(B_raw, allow_missing=True)

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._component_specs = specs
        self._basis_train = np.asarray(B, dtype=np.float64)
        self._record_constraint_result(None, None, absorbed_by=None)

        self._tensor_basis_raw = B_raw
        q = self._basis_train.shape[1]

        if self.xt is None or self.xt.get("S", None) is None:
            S_raw = np.eye(q, dtype=np.float64)
            self._tensor_penalties_raw = [S_raw]
            self._penalties = [scale_penalty(B_raw, S_raw)]
            self._ranks = [q]
            self._set_penalty_rescale_factors([penalty_rescale_factor(B_raw, S_raw)])
        else:
            S_in = self.xt["S"]
            S_list = S_in if isinstance(S_in, list) else [S_in]
            S_list = [np.asarray(S, dtype=np.float64) for S in S_list]

            for S in S_list:
                if S.shape != (q, q):
                    raise ValueError(
                        f"Supplied random-effect penalty has shape {S.shape}, expected {(q, q)}."
                    )

            ranks = self.xt.get("rank", None)
            if ranks is None:
                raise ValueError(
                    "For bs='re' with xt['S'], xt must also supply 'rank', matching mgcv."
                )
            ranks = np.asarray(ranks, dtype=int).ravel()
            if ranks.shape != (len(S_list),):
                raise ValueError(
                    f"xt['rank'] must have shape ({len(S_list)},), got {ranks.shape}."
                )

            S_raw_list = [symmetrize_matrix(S) for S in S_list]
            self._tensor_penalties_raw = S_raw_list
            self._penalties = [scale_penalty(B_raw, S_raw) for S_raw in S_raw_list]
            self._ranks = [int(r) for r in ranks.tolist()]
            self._set_penalty_rescale_factors(
                [penalty_rescale_factor(B_raw, S_raw) for S_raw in S_raw_list]
            )

        return self

    @property
    def zero_unseen_factor_features(self):
        return tuple(
            name
            for name, component in zip(
                self._feature_names or (), self._component_specs or (), strict=True
            )
            if component.kind == "factor"
        )

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del centered, apply_np, x_train
        self._require_fitted()
        if len(self._tensor_penalties_raw) != 1:
            raise NotImplementedError(
                "Tensor products of smooths with multiple penalties are not supported."
            )
        return self._tensor_basis_raw, self._tensor_penalties_raw[0], None

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

        raw = list(self.penalties)
        sp_vals = self._normalized_term_sp(len(raw))
        defs = []

        for j, P in enumerate(raw):
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            if sp_j is None:
                sp_mode = None
                sp_value = None
            elif sp_j >= 0:
                sp_mode = "fixed"
                sp_value = float(sp_j)
            else:
                sp_mode = "estimate"
                sp_value = None

            defs.append(
                PenaltySpec(
                    matrix=np.asarray(P, dtype=np.float64),
                    smoothing_id=None,
                    kind="random_effect",
                    rank=self._ranks[j] if self._ranks is not None else None,
                    null_space_dim=0,
                    is_null_space_penalty=False,
                    sp_mode=sp_mode,
                    sp_value=sp_value,
                    metadata=self._penalty_metadata_with_scale(
                        {
                            "term_type": self.term_type,
                            "basis_name": self.basis_name,
                            "feature": list(self.feature),
                            "label": self.label,
                            "by": self.by,
                            "by_name": self._by_state.feature_name,
                            "xt": self.xt,
                            "component_specs": [
                                {
                                    "kind": s.kind,
                                    "levels": (
                                        None if s.levels is None else list(s.levels)
                                    ),
                                }
                                for s in (self._component_specs or [])
                            ],
                        },
                        penalty_index=j,
                    ),
                )
            )

            if self.select:
                selection_meta = {
                    "term_type": self.term_type,
                    "basis_name": self.basis_name,
                    "feature": list(self.feature),
                    "label": self.label,
                    "by": self.by,
                    "by_name": self._by_state.feature_name,
                    "is_selection_penalty": True,
                    "xt": self.xt,
                    "component_specs": [
                        {
                            "kind": s.kind,
                            "levels": None if s.levels is None else list(s.levels),
                        }
                        for s in (self._component_specs or [])
                    ],
                }
                defs.extend(
                    self._build_selection_penalty_definitions(
                        [np.asarray(P, dtype=np.float64)],
                        selection_metadata=selection_meta,
                    )
                )

        return defs

    def transform_new(self, X_new):
        self._require_fitted()

        blocks = []
        columns = functional_object_columns(X_new, self._feature_indices)
        for column, spec in zip(columns, self._component_specs, strict=True):
            blocks.append(_predict_re_component(column.reshape(-1), spec))

        B = _combine_random_effect_blocks(blocks)

        z = by_values_from_new_data(X_new, self._by_state)
        if columns[0].ndim == 2:
            return aggregate_functional_basis(B, columns[0].shape, z)
        return np.asarray(
            self._apply_by_scale(B, z, allow_missing=True), dtype=np.float64
        )
