"""
Tensor-product smooth term (``bs='te'``).

Implements the :class:`BaseSmoothTerm` interface for a full interaction tensor
product of marginal spline bases.  The penalty is a sum of Kronecker-product
marginal penalties, each penalising the corresponding marginal function.

The ``te`` smooth is the standard interaction smooth: it adds a penalty for
each axis and does not impose an ANOVA decomposition. Use ``ti`` for
main-effects-removed interaction terms.
"""

import numpy as np

from ...constraints.absorption import full_term_sum_to_zero_constraint
from ...penalties import (
    penalty_id_for_local_index,
    rescale_tensor_penalties_for_fit,
    tensor_product_penalties,
)
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    build_penalty_definition,
)
from .marginals import (
    build_tensor_marginal_terms,
    build_tensor_product_components,
    normalize_tensor_fx_flags,
    resolve_tensor_marginal_features,
    tensor_predict_matrix,
    validate_tensor_marginal_bases,
)


@register_smooth("te")
class TensorProductSplineTerm(BaseSmoothTerm):
    term_type = "tensor_smooth"
    basis_name = "te"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=10,
        basis="cr",
        m=None,
        xt=None,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        null_penalty_tol=1e-10,
        knots=None,
        pc=None,
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
        self.m = m
        self.xt = xt

        self.select = bool(select)
        self.fixed_flags = normalize_tensor_fx_flags(
            fixed,
            len(features),
        )
        self.fixed = bool(all(self.fixed_flags))
        self.null_penalty_tol = float(null_penalty_tol)
        self.knots = _normalize_knots(knots, features)
        self.pc = pc

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._marginal_np_transforms = None
        self._basis_train = None
        self._penalties = None
        self._basis_dims = None

        self._by_state = None

    def fit(self, X, feature_names):
        marginal_shared_setups = self._linked_id_marginal_setups(self.feature)
        marginals, _, _ = build_tensor_marginal_terms(
            feature=self.feature,
            k=self.k,
            basis=self.basis,
            m=self.m,
            xt=self.xt,
            knots=self.knots,
            centered=False,
            shared_basis_setups=marginal_shared_setups,
        )
        for term in marginals:
            term.fit(X, feature_names)
        feature_indices, feature_names_resolved = resolve_tensor_marginal_features(
            marginals
        )

        self._set_by_state(X, feature_names)

        # Build marginal bases and penalties matching mgcv's te construction:
        # 1. Use unscaled raw penalty (no scale_penalty yet).
        # 2. Apply np=TRUE conditioning transform (SVD-based orthogonalisation).
        # 3. Eigenvalue-normalize each marginal penalty.
        # 4. Build tensor product.
        # 5. Apply outer scale_penalty to the full tensor basis and penalties.
        (
            _marginal_bases,
            marginal_penalties,
            marginal_np_transforms,
            basis_dims,
            B_raw,
            B_setup,
        ) = build_tensor_product_components(
            marginals,
            X,
            use_centered=[False] * len(marginals),
            apply_np=True,
        )
        S_raw = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)

        # Outer scale_penalty on the full tensor product (matches smoothCon outer step).
        S_raw, penalty_scales = rescale_tensor_penalties_for_fit(
            B_setup, S_raw, return_scales=True
        )

        factor_by_meta = (
            isinstance(self.metadata, dict)
            and self.metadata.get("factor_by", None) is not None
        )
        if self.pc is not None:
            max_index = max(feature_indices)

            def point_basis_fn(point):
                point_data = np.zeros((point.shape[0], max_index + 1), dtype=np.float64)
                point_data[:, feature_indices] = point
                return tensor_predict_matrix(
                    marginals,
                    point_data,
                    centered=False,
                    np_transforms=marginal_np_transforms,
                )

            B_te, S_te, C_te, _ = self._apply_point_constraint(
                B_raw,
                S_raw,
                self.pc,
                feature_names=feature_names_resolved,
                point_basis_fn=point_basis_fn,
                fixed=False,
            )
            constraint_kind = "pc"
        elif self._by_state.is_constant or factor_by_meta:
            if self._linked_id_setup() is None:
                B_te, S_te, C_te = full_term_sum_to_zero_constraint(B_raw, S_raw)
            else:
                _, S_te, C_te = full_term_sum_to_zero_constraint(B_setup, S_raw)
                B_te = B_raw @ C_te
        else:
            B_te, S_te, C_te = B_raw, S_raw, None
            constraint_kind = None

        if self.pc is None:
            B_te = self._apply_cached_by(B_te)
            if C_te is not None:
                constraint_kind = "sum_to_zero"

        self._marginals = marginals
        self._marginal_np_transforms = marginal_np_transforms
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._basis_dims = basis_dims
        self._basis_train = np.asarray(B_te, dtype=np.float64)
        keep_penalties = [not flag for flag in self.fixed_flags]
        self._penalties = [
            np.asarray(S, dtype=np.float64)
            for S, keep in zip(S_te, keep_penalties, strict=True)
            if keep
        ]
        self._set_penalty_rescale_factors(
            [
                float(scale)
                for scale, keep in zip(
                    penalty_scales, keep_penalties, strict=True
                )
                if keep
            ]
        )
        self._record_constraint_result(
            constraint_kind,
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
                else penalty_id_for_local_index(self.smoothing_id, j, n_penalties=n_raw)
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
                    local_penalty_index=j,
                )
            )

        defs.extend(
            self._build_selection_penalty_definitions(
                raw,
                null_penalty_tol=self.null_penalty_tol,
            )
        )

        return defs

    def transform_new(self, X_new):
        self._require_fitted()

        B_new_raw = tensor_predict_matrix(
            self._marginals,
            X_new,
            centered=False,
            np_transforms=self._marginal_np_transforms,
        )
        return self._apply_constraint_transform_and_by(B_new_raw, X_new)
