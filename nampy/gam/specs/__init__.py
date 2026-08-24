"""Canonical declarative GAM specification objects."""

from .base import TermSpec
from .build import build_smooth_spec, smooth_spec_from_basis_options
from .predictors import LinearPredictorSpec, PenaltyGroupSpec
from .smooth import (
    BaseSmoothSpec,
    CubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    DerivativeBSplineSmoothSpec,
    FactorSmoothInteractionSpec,
    PSplineSmoothSpec,
    RandomEffectSmoothSpec,
    ShapeConstrainedSmoothSpec,
    SmoothSpec,
    SumToZeroFactorSmoothSpec,
    TensorInteractionSmoothSpec,
    TensorProductSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    ThinPlateSmoothSpec,
    replace_smooth_spec,
)

__all__ = [
    "TermSpec",
    "LinearPredictorSpec",
    "PenaltyGroupSpec",
    "BaseSmoothSpec",
    "CubicRegressionSmoothSpec",
    "CubicShrinkageSmoothSpec",
    "CyclicCubicRegressionSmoothSpec",
    "DerivativeBSplineSmoothSpec",
    "FactorSmoothInteractionSpec",
    "PSplineSmoothSpec",
    "RandomEffectSmoothSpec",
    "ShapeConstrainedSmoothSpec",
    "SmoothSpec",
    "SumToZeroFactorSmoothSpec",
    "TensorInteractionSmoothSpec",
    "TensorProductSmoothSpec",
    "ThinPlateShrinkageSmoothSpec",
    "ThinPlateSmoothSpec",
    "build_smooth_spec",
    "smooth_spec_from_basis_options",
    "replace_smooth_spec",
]
