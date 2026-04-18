"""
Interaction tensor-product smooth with main effects removed (``bs='ti'``).

Like ``te``, but marginal main-effect columns are projected out so that
the resulting term represents a pure interaction.  Best used alongside
separate ``s()`` terms for each marginal, decomposing the full interaction
``te(x1, x2)`` = ``s(x1) + s(x2) + ti(x1, x2)``.
"""

import warnings

import numpy as np

from ...penalties import (
    penalty_id_for_local_index,
    rescale_tensor_penalties_for_fit,
    tensor_product_penalties,
)
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    _normalize_mc,
    build_penalty_definition,
    by_values_from_new_data,
)
from .marginals import (
    build_tensor_marginal_terms,
    build_tensor_product_components,
    resolve_tensor_marginal_features,
    tensor_predict_matrix,
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
        m=None,
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
        self.m = m

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
        self._set_by_state(X, feature_names)

        self._mc = _normalize_mc(self.mc, len(self.feature))
        marginal_shared_setups = self._linked_id_marginal_setups()
        marginals, _, _ = build_tensor_marginal_terms(
            feature=self.feature,
            k=self.k,
            basis=self.basis,
            m=self.m,
            knots=self.knots,
            centered=self._mc,
            shared_basis_setups=marginal_shared_setups,
        )
        for term in marginals:
            term.fit(X, feature_names)
        feature_indices, feature_names_resolved = resolve_tensor_marginal_features(
            marginals
        )

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

        # Build marginal bases and penalties matching mgcv's ti construction.
        (
            _marginal_bases,
            marginal_penalties,
            marginal_np_transforms,
            basis_dims,
            B_ti_raw,
            B_ti_setup,
        ) = build_tensor_product_components(
            marginals,
            X,
            use_centered=use_centered,
            apply_np=True,
        )
        S_ti = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)

        # Outer scale_penalty on the tensor product (matches smoothCon outer step).
        S_ti = rescale_tensor_penalties_for_fit(B_ti_setup, S_ti)

        B_ti = self._apply_cached_by(B_ti_raw)

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
                )
            )

        defs.extend(
            self._build_selection_penalty_definitions(
                raw,
                null_penalty_tol=self.null_penalty_tol,
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
        B = tensor_predict_matrix(
            self._marginals,
            X_new,
            centered=self._marginal_is_centered,
            np_transforms=self._marginal_np_transforms,
        )
        z = by_values_from_new_data(X_new, self._by_state)
        return self._apply_by_scale(B, z)
