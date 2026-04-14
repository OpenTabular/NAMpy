"""
Interaction tensor-product smooth with main effects removed (``bs='ti'``).

Like ``te``, but marginal main-effect columns are projected out so that
the resulting term represents a pure interaction.  Best used alongside
separate ``s()`` terms for each marginal, decomposing the full interaction
``te(x1, x2)`` = ``s(x1) + s(x2) + ti(x1, x2)``.
"""

import warnings

import numpy as np

from ...basis.algebra import rowwise_kronecker
from ...penalties import (
    normalize_tensor_marginal_penalty,
    rescale_tensor_penalties_for_fit,
    tensor_product_penalties,
)
from ...penalties.algebra import null_space_penalty_from_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    _normalize_mc,
    build_penalty_definition,
    build_selection_penalty_definition,
    by_values_from_new_data,
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


@register_smooth("ti")
class InteractionTensorProductSplineTerm(BaseSmoothTerm):
    term_type = "tensor_interaction"
    basis_name = "ti"
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
        mc=None,
        select=False,
        fixed=False,
        null_penalty_tol=1e-10,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 1:
            raise ValueError(
                "InteractionTensorProductSplineTerm requires at least one feature."
            )

        super().__init__(
            feature=features,
            label=label or f"ti({', '.join(map(str, features))})",
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

        self.mc = mc
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.null_penalty_tol = float(null_penalty_tol)
        self.knots = _normalize_knots(knots, features)

        self._mc = None
        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._marginal_np_transforms = None
        self._basis_train = None
        self._penalties = None
        self._basis_dims = None
        self._marginal_is_centered = None

        self._by_state = None

    def fit(self, X, feature_names):
        marginals = []
        feature_indices = []
        feature_names_resolved = []

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        self._mc = _normalize_mc(self.mc, len(self.feature))

        for feat, k_i, bs_i, knots_i, center_i in zip(
            self.feature, self.k, self.basis, self.knots, self._mc
        ):
            term = make_tensor_marginal_term(
                feature=feat,
                basis=bs_i,
                k=k_i,
                knots=knots_i,
                centered=bool(center_i),
            )
            term.fit(X, feature_names)
            marginals.append(term)
            feature_indices.append(tensor_marginal_feature_index(term))
            feature_names_resolved.append(tensor_marginal_feature_name(term))

        marginal_bases = []
        marginal_penalties = []
        basis_dims = []
        marginal_np_transforms = []

        if self.by is not None and not self._by_state.is_constant:
            if self.mc is not None:
                warnings.warn(
                    f"{self.label}: ignoring mc={self.mc} because numeric by={self._by_state.feature_name!r} "
                    "is non-constant, so automatic identifiability constraints are not applied.",
                    stacklevel=2,
                )
            use_centered = [False] * len(marginals)
        else:
            use_centered = list(self._mc)

        # Build marginal bases and penalties matching mgcv's ti construction:
        # - mc=TRUE marginals: smoothCon basis (scale_penalty + constraint absorbed),
        #   np conditioning via the centered prediction, eigenvalue normalize.
        # - mc=FALSE marginals: raw (unscaled) basis, np conditioning via raw prediction,
        #   eigenvalue normalize.
        # After the tensor product, apply outer scale_penalty (smoothCon outer step).
        for m, center_this_margin in zip(marginals, use_centered):
            x_train = column_as_float(X, tensor_marginal_feature_index(m))
            B, S, XP = tensor_marginal_fit_matrices(
                m,
                centered=bool(center_this_margin),
                apply_np=True,
                x_train=x_train,
            )
            S = normalize_tensor_marginal_penalty(S)
            marginal_bases.append(B)
            marginal_penalties.append(S)
            basis_dims.append(B.shape[1])
            marginal_np_transforms.append(XP)

        B_ti_raw = rowwise_kronecker(marginal_bases)
        S_ti = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)

        # Outer scale_penalty on the tensor product (matches smoothCon outer step).
        S_ti = rescale_tensor_penalties_for_fit(B_ti_raw, S_ti)

        if self._by_state.is_present:
            B_ti = B_ti_raw * self._by_state.values[:, None]
        else:
            B_ti = B_ti_raw

        self._marginals = marginals
        self._marginal_np_transforms = marginal_np_transforms
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._basis_dims = basis_dims
        self._marginal_is_centered = list(use_centered)
        self._basis_train = np.asarray(B_ti, dtype=np.float64)
        self._penalties = [] if self.fixed else S_ti
        self._record_constraint_result(None, None, absorbed_by=None)

        self.basis_name = "ti(" + ",".join(self.basis) + ")"
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
        self._require_fitted()

        marginal_new = []
        for m, center_this_margin, xp in zip(
            self._marginals, self._marginal_is_centered, self._marginal_np_transforms
        ):
            marginal_new.append(
                tensor_marginal_predict_matrix(
                    m, X_new, centered=bool(center_this_margin), np_transform=xp
                )
            )

        B = rowwise_kronecker(marginal_new)
        z = by_values_from_new_data(X_new, self._by_state)
        if z is not None:
            B = B * z[:, None]
        return B
