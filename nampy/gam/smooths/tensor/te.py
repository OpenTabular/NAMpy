"""
Tensor-product smooth term (``bs='te'``).

Implements the :class:`BaseSmoothTerm` interface for a full interaction tensor
product of marginal spline bases.  The penalty is a sum of Kronecker-product
marginal penalties, each penalising the corresponding marginal function.

The ``te`` smooth is the standard interaction smooth: it adds a penalty for
each axis and does not impose an ANOVA decomposition.  Use ``ti`` for
main-effects-removed interaction terms or ``t2`` for the alternative
penalty parameterisation.
"""

import numpy as np

from ..base import (
    BaseSmoothTerm,
    _normalize_knots,
    column_as_float,
    resolve_by_state,
    sync_by_state_attributes,
)
from ...constraints.absorption import full_term_sum_to_zero_constraint
from ..registry import register_smooth
from ..univariate.cubic_regression import SplineTerm1D
from ...basis.tensor import rowwise_kronecker, tensor_product_penalties


@register_smooth("te")
@register_smooth("tensor")
class TensorProductSplineTerm(BaseSmoothTerm):
    term_type = "tensor_smooth"
    basis_name = "te"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=10,
        basis="cr",
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 2:
            raise ValueError("TensorProductSplineTerm requires at least two features.")

        super().__init__(
            feature=features,
            label=label or f"te({', '.join(map(str, features))})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        if np.isscalar(k):
            self.k = [int(k)] * len(features)
        else:
            self.k = [int(v) for v in k]
        if len(self.k) != len(features):
            raise ValueError(
                f"k must have length {len(features)} for features={features}, got {self.k}."
            )

        if isinstance(basis, str):
            self.basis = [basis] * len(features)
        else:
            self.basis = [str(v) for v in basis]
        if len(self.basis) != len(features):
            raise ValueError(
                f"basis must have length {len(features)} for features={features}, got {self.basis}."
            )

        if any(bs != "cr" for bs in self.basis):
            raise NotImplementedError(
                "TensorProductSplineTerm currently supports only basis='cr' marginals."
            )

        self.fixed = bool(fixed)
        self.knots = _normalize_knots(knots, features)

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._basis_train = None
        self._penalties = None
        self._basis_dims = None
        self._constraint_transform = None

        self._by_state = None

    def fit(self, X, feature_names):
        marginals = []
        feature_indices = []
        feature_names_resolved = []

        for feat, k_i, bs_i, knots_i in zip(self.feature, self.k, self.basis, self.knots):
            term = SplineTerm1D(
                feature=feat,
                k=k_i,
                basis=bs_i,
                label=str(feat),
                smoothing_id=None,
                by=None,
                select=False,
                fixed=False,
                knots=knots_i,
            )
            term.fit(X, feature_names)
            marginals.append(term)
            feature_indices.append(term._feature_index)
            feature_names_resolved.append(term._feature_name)

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        raw_marginal_bases = [m._spline.raw_basis for m in marginals]
        raw_marginal_penalties = [m._spline.raw_penalty for m in marginals]
        basis_dims = [B.shape[1] for B in raw_marginal_bases]

        B_raw = rowwise_kronecker(raw_marginal_bases)
        S_raw = tensor_product_penalties(raw_marginal_penalties, basis_dims=basis_dims)

        if self._by_state.is_constant:
            B_te, S_te, C_te = full_term_sum_to_zero_constraint(B_raw, S_raw)
        else:
            B_te, S_te, C_te = B_raw, S_raw, None

        if self._by_state.is_present:
            B_te = B_te * self._by_state.values[:, None]

        self._marginals = marginals
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._basis_dims = basis_dims
        self._basis_train = np.asarray(B_te, dtype=np.float64)
        self._penalties = [] if self.fixed else [np.asarray(S, dtype=np.float64) for S in S_te]
        self._constraint_transform = C_te
        self._record_constraint_result(
            "sum_to_zero" if C_te is not None else None,
            C_te,
            absorbed_by=("runtime" if C_te is not None else None),
        )

        self.basis_name = "te(" + ",".join(self.basis) + ")"
        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._penalties is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def transform_new(self, X_new):
        if self._marginals is None:
            raise RuntimeError("Term is not fitted.")

        raw_marginal_new = [
            m._spline.transform_new_raw(column_as_float(X_new, m._feature_index))
            for m in self._marginals
        ]
        B_new_raw = rowwise_kronecker(raw_marginal_new)
        return self._apply_constraint_transform_and_by(B_new_raw, X_new)
