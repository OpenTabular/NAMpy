"""Spherical spline smooth term (``bs='sos'``)."""

from __future__ import annotations

from ...splines.univariate.sos import (
    build_spherical_spline_setup,
    predict_spherical_spline,
)
from ..registry import register_smooth
from ._single_penalty import SinglePenaltyLowRankSmoothTerm


@register_smooth("sos")
class SphericalSplineTerm(SinglePenaltyLowRankSmoothTerm):
    term_type = "smooth"
    basis_name = "sos"

    def _validate_features(self, features) -> None:
        if len(features) != 2:
            raise ValueError(
                "Can only deal with a sphere: bs='sos' requires exactly two "
                "features, latitude first and longitude second."
            )

    def _build_setup(self, values):
        return build_spherical_spline_setup(
            values,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )

    def _predict_raw(self, values):
        return predict_spherical_spline(values, self._setup)

    def _basis_metadata(self) -> dict:
        return {
            "spherical_order": int(self._setup.order),
            "original_null_space_dim": int(self._setup.null_space_dim),
        }

    def validate_factor_smooth_base(self, mode: str) -> None:
        self._require_fitted()
        if str(mode).lower() == "fs" and int(self._setup.null_space_dim) > 1:
            raise NotImplementedError(
                "bs='fs' with an SOS m=-1 base is not enabled: mgcv's four-way "
                "repeated null eigenspace receives separate penalties whose "
                "orientation is LAPACK-dependent. Use another SOS order."
            )


__all__ = ["SphericalSplineTerm"]
