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

from ...basis.tensor import (
    normalize_tensor_marginal_penalty,
    rescale_tensor_penalties_for_fit,
    rowwise_kronecker,
    tensor_product_penalties,
)
from ...constraints.absorption import full_term_sum_to_zero_constraint
from ...penalties.algebra import null_space_penalty_from_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    build_penalty_definition,
    build_selection_penalty_definition,
    column_as_float,
    resolve_by_state,
    sync_by_state_attributes,
)
from .marginals import (
    make_tensor_marginal_term,
    tensor_marginal_feature_index,
    tensor_marginal_feature_name,
    tensor_marginal_fit_matrices,
    tensor_marginal_predict_matrix,
    validate_tensor_marginal_bases,
)


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
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        null_penalty_tol=1e-10,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 2:
            raise ValueError("TensorProductSplineTerm requires at least two features.")

        super().__init__(
            feature=features,
            label=label or f"te({', '.join(map(str, features))})",
            term_id=term_id,
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
        self.basis = validate_tensor_marginal_bases(self.basis)

        self.select = bool(select)
        self.fixed = bool(fixed)
        self.null_penalty_tol = float(null_penalty_tol)
        self.knots = _normalize_knots(knots, features)

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._marginal_np_transforms = None
        self._basis_train = None
        self._penalties = None
        self._basis_dims = None

        self._by_state = None

    def fit(self, X, feature_names):
        marginals = []
        feature_indices = []
        feature_names_resolved = []

        for feat, k_i, bs_i, knots_i in zip(
            self.feature, self.k, self.basis, self.knots
        ):
            term = make_tensor_marginal_term(
                feature=feat,
                basis=bs_i,
                k=k_i,
                knots=knots_i,
                centered=False,
            )
            term.fit(X, feature_names)
            marginals.append(term)
            feature_indices.append(tensor_marginal_feature_index(term))
            feature_names_resolved.append(tensor_marginal_feature_name(term))

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        # Build marginal bases and penalties matching mgcv's te construction:
        # 1. Use unscaled raw penalty (no scale_penalty yet).
        # 2. Apply np=TRUE conditioning transform (SVD-based orthogonalisation).
        # 3. Eigenvalue-normalize each marginal penalty.
        # 4. Build tensor product.
        # 5. Apply outer scale_penalty to the full tensor basis and penalties.
        marginal_bases = []
        marginal_penalties = []
        marginal_np_transforms = []
        for m in marginals:
            x_train = column_as_float(X, tensor_marginal_feature_index(m))
            X_j, S_j, XP_j = tensor_marginal_fit_matrices(
                m, centered=False, apply_np=True, x_train=x_train
            )
            S_j = normalize_tensor_marginal_penalty(S_j)
            marginal_bases.append(X_j)
            marginal_penalties.append(S_j)
            marginal_np_transforms.append(XP_j)

        basis_dims = [B.shape[1] for B in marginal_bases]

        B_raw = rowwise_kronecker(marginal_bases)
        S_raw = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)

        # Outer scale_penalty on the full tensor product (matches smoothCon outer step).
        S_raw = rescale_tensor_penalties_for_fit(B_raw, S_raw)

        if self._by_state.is_constant:
            B_te, S_te, C_te = full_term_sum_to_zero_constraint(B_raw, S_raw)
        else:
            B_te, S_te, C_te = B_raw, S_raw, None

        if self._by_state.is_present:
            B_te = B_te * self._by_state.values[:, None]

        self._marginals = marginals
        self._marginal_np_transforms = marginal_np_transforms
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._basis_dims = basis_dims
        self._basis_train = np.asarray(B_te, dtype=np.float64)
        self._penalties = (
            [] if self.fixed else [np.asarray(S, dtype=np.float64) for S in S_te]
        )
        self._record_constraint_result(
            "sum_to_zero" if C_te is not None else None,
            C_te,
            absorbed_by=("runtime" if C_te is not None else None),
        )

        self.basis_name = "te(" + ",".join(self.basis) + ")"
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        raw = list(self.penalties)
        if len(raw) == 0:
            return []

        n_raw = len(raw)
        sp_vals = self._normalized_term_sp(n_raw)
        defs = []
        for j, P in enumerate(raw):
            sid = (
                None
                if self.smoothing_id is None
                else (
                    str(self.smoothing_id)
                    if n_raw <= 1
                    else f"{self.smoothing_id}::{j}"
                )
            )
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            defs.append(
                build_penalty_definition(
                    self,
                    P,
                    kind="smooth",
                    smoothing_id=sid,
                    sp_value_in=sp_j,
                    metadata_extra={"term_sp": sp_j, "is_selection_penalty": False},
                )
            )

        if self.select:
            combined = sum(np.asarray(P, dtype=np.float64) for P in raw)
            S0, meta = null_space_penalty_from_penalty(
                combined, tol=self.null_penalty_tol
            )
            if meta["rank"] > 0:
                select_sid = (
                    None
                    if self.smoothing_id is None
                    else f"{self.smoothing_id}::select"
                )
                defs.append(
                    build_selection_penalty_definition(
                        self,
                        S0,
                        rank=meta["rank"],
                        null_space_dim=meta["null_space_dim"],
                        smoothing_id=select_sid,
                    )
                )

        return defs

    def transform_new(self, X_new):
        self._require_fitted()

        marginal_new = []
        for m, xp in zip(self._marginals, self._marginal_np_transforms):
            marginal_new.append(
                tensor_marginal_predict_matrix(
                    m, X_new, centered=False, np_transform=xp
                )
            )
        B_new_raw = rowwise_kronecker(marginal_new)
        return self._apply_constraint_transform_and_by(B_new_raw, X_new)
