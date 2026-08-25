"""Gaussian-process smooth term (``bs='gp'``)."""

from __future__ import annotations

import numpy as np

from ...splines.univariate.gp import (
    build_gaussian_process_setup,
    predict_gaussian_process,
)
from ..registry import register_smooth
from ._single_penalty import SinglePenaltyLowRankSmoothTerm


@register_smooth("gp")
class GaussianProcessTerm(SinglePenaltyLowRankSmoothTerm):
    term_type = "smooth"
    basis_name = "gp"

    def _build_setup(self, values):
        return build_gaussian_process_setup(
            values,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )

    def _predict_raw(self, values):
        return predict_gaussian_process(values, self._setup)

    def _basis_metadata(self) -> dict:
        return {"gp_definition": np.asarray(self._setup.definition).tolist()}


__all__ = ["GaussianProcessTerm"]
