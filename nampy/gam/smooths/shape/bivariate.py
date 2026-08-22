"""Runtime terms for SCAM's bivariate shape-constrained P-splines."""

from __future__ import annotations

import numpy as np

from ...penalties import penalty_id_for_local_index
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.shape import build_bivariate_shape_setup, predict_bivariate_shape
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    build_penalty_definition,
    column_as_float,
)


@register_smooth("tedmi")
@register_smooth("tedmd")
@register_smooth("temicx")
@register_smooth("temicv")
@register_smooth("tedecv")
@register_smooth("tedecx")
@register_smooth("tecvcv")
@register_smooth("tecxcx")
@register_smooth("tecxcv")
@register_smooth("tescv")
@register_smooth("tescx")
@register_smooth("tesmi1")
@register_smooth("tesmd1")
@register_smooth("tesmi2")
@register_smooth("tesmd2")
@register_smooth("tismi")
@register_smooth("tismd")
class BivariateShapePSplineTerm(BaseSmoothTerm):
    """Two-dimensional SCAM tensor-product SCOP spline."""

    term_type = "bivariate_shape_constrained_smooth"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        *,
        k=7,
        basis="tedmi",
        m=None,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        knots=None,
        metadata=None,
        **_ignored,
    ):
        features = list(feature) if isinstance(feature, (list, tuple)) else [feature]
        if len(features) != 2:
            raise ValueError("Bivariate SCAM smooths require exactly two features.")
        super().__init__(
            feature=features,
            label=label,
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.metadata["coefficient_covariance_transport"] = "prediction"
        if by is not None:
            raise NotImplementedError("Bivariate SCAM by-variable terms are not supported.")
        if select:
            raise NotImplementedError("SCAM does not expose select= for bivariate bases.")
        self.features = features
        self.k = k
        self.m = m
        self.basis_name = str(basis).lower()
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.knots = knots
        self._feature_indices = None
        self._feature_names = None
        self._setup = None
        self._basis_train = None
        self._penalties = None
        self.positive_coefficient_mask = None

    def fit(self, X, feature_names):
        resolved = [_resolve_feature(feature, feature_names) for feature in self.features]
        self._feature_indices = [item[0] for item in resolved]
        self._feature_names = [item[1] for item in resolved]
        self._set_resolved_features(self._feature_names)
        x = column_as_float(X, self._feature_indices[0])
        z = column_as_float(X, self._feature_indices[1])
        self._setup = build_bivariate_shape_setup(
            x,
            z,
            basis_code=self.basis_name,
            bs_dim=self.k,
            spline_order=self.m,
            knots=self.knots,
        )
        self._basis_train = np.asarray(self._setup.basis_train, dtype=np.float64)
        scales = [
            penalty_rescale_factor(self._basis_train, penalty)
            for penalty in self._setup.penalties
        ]
        self._set_penalty_rescale_factors(scales)
        self._penalties = (
            []
            if self.fixed
            else [
                scale_penalty(self._basis_train, penalty)
                for penalty in self._setup.penalties
            ]
        )
        self.positive_coefficient_mask = self._setup.positive_mask.copy()
        self._record_constraint_result(
            "scam_constructor_centering", None, absorbed_by="runtime"
        )
        return self

    def get_penalty_definitions(self):
        self._require_fitted()
        penalties = list(self.penalties)
        if not penalties:
            return []
        sp_values = self._normalized_term_sp(len(penalties))
        definitions = []
        for index, penalty in enumerate(penalties):
            smoothing_id = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(
                    self.smoothing_id, index, n_penalties=len(penalties)
                )
            )
            sp_value = sp_values[index]
            definitions.append(
                build_penalty_definition(
                    self,
                    penalty,
                    smoothing_id=smoothing_id,
                    sp_value_in=sp_value,
                    rank=self._setup.ranks[index],
                    null_space_dim=self._setup.null_space_dim,
                    metadata_extra={
                        "basis_name": self.basis_name,
                        "shape_constraint": self.basis_name,
                        "p_ident": self.positive_coefficient_mask.tolist(),
                        "term_sp": sp_value,
                    },
                    local_penalty_index=index,
                )
            )
        return definitions

    def transform_new(self, X_new):
        self._require_fitted()
        x = column_as_float(X_new, self._feature_indices[0])
        z = column_as_float(X_new, self._feature_indices[1])
        prediction = predict_bivariate_shape(x, z, self._setup)
        return prediction @ self._setup.constraint_matrix


__all__ = ["BivariateShapePSplineTerm"]
