"""Duchon regression-spline smooth term (``bs='ds'``)."""

from __future__ import annotations

from ...splines.univariate.ds import (
    build_duchon_spline_setup,
    predict_duchon_spline,
)
from ..registry import register_smooth
from ._single_penalty import SinglePenaltyLowRankSmoothTerm


@register_smooth("ds")
class DuchonSplineTerm(SinglePenaltyLowRankSmoothTerm):
    term_type = "smooth"
    basis_name = "ds"

    def _build_setup(self, values):
        return build_duchon_spline_setup(
            values,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )

    def _predict_raw(self, values):
        return predict_duchon_spline(values, self._setup)

    def _basis_metadata(self) -> dict:
        return {
            "penalty_order": self._setup.penalty_order,
            "shift_order": self._setup.shift_order,
        }


__all__ = ["DuchonSplineTerm"]
