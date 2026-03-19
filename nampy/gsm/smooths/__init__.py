from .registry import register_smooth, available_smooths, make_smooth_term
from .base import (
    BaseSmoothTerm,
    _resolve_feature,
    _resolve_numeric_by,
    _normalize_knots,
    _apply_sum_to_zero_constraint,
    _is_effectively_constant,
    _full_term_sum_to_zero_constraint,
    _normalize_mc,
    _normalize_point_constraint,
    _apply_linear_constraint,
)

from .constructed import ConstructedSmooth, predict_mat

from .univariate.cubic_regression import LinearTerm, SplineTerm1D
from .univariate.pspline import PSplineTerm1D
from .univariate.thin_plate import ThinPlateSplineTerm
from .univariate.gp import GPSmoothTerm
from .tensor.te import TensorProductSplineTerm
from .tensor.ti import InteractionTensorProductSplineTerm
from .tensor.t2 import TensorANOVASplineTerm

__all__ = [
    "register_smooth",
    "available_smooths",
    "make_smooth_term",
    "BaseSmoothTerm",
    "_resolve_feature",
    "_resolve_numeric_by",
    "_normalize_knots",
    "_apply_sum_to_zero_constraint",
    "_is_effectively_constant",
    "_full_term_sum_to_zero_constraint",
    "_normalize_mc",
    "_normalize_point_constraint",
    "_apply_linear_constraint",
    "ConstructedSmooth",
    "predict_mat",
    "LinearTerm",
    "SplineTerm1D",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
    "GPSmoothTerm",
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "TensorANOVASplineTerm",
]
