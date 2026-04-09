# After univariate/tensor smooths load (SplineTerm1D, etc.); registers parametric "linear".
from ..terms.linear import LinearTerm as _LinearTerm  # noqa: F401
from .base import (
    RUNTIME_TERM_INTERFACE_CHECKLIST,
    BaseSmoothTerm,
    ByState,
    _is_effectively_constant,
    _normalize_knots,
    _normalize_mc,
    _normalize_point_constraint,
    _resolve_feature,
    _resolve_numeric_by,
    apply_numeric_by,
    build_penalty_definition,
    build_selection_penalty_definition,
    by_values_from_new_data,
    column_as_float,
    column_as_object,
    columns_as_float_matrix,
    resolve_by_state,
    resolve_feature_matrix_state,
    sync_by_state_attributes,
)
from .registry import available_smooths, make_smooth_term, register_smooth
from .tensor.t2 import TensorANOVASplineTerm
from .tensor.te import TensorProductSplineTerm
from .tensor.ti import InteractionTensorProductSplineTerm
from .univariate.cubic_regression import SplineTerm1D
from .univariate.gp import GPSmoothTerm
from .univariate.pspline import PSplineTerm1D
from .univariate.thin_plate import ThinPlateSplineTerm

__all__ = [
    "register_smooth",
    "available_smooths",
    "make_smooth_term",
    "BaseSmoothTerm",
    "ByState",
    "RUNTIME_TERM_INTERFACE_CHECKLIST",
    "_resolve_feature",
    "_resolve_numeric_by",
    "_normalize_knots",
    "_is_effectively_constant",
    "_normalize_mc",
    "_normalize_point_constraint",
    "apply_numeric_by",
    "by_values_from_new_data",
    "column_as_float",
    "column_as_object",
    "columns_as_float_matrix",
    "resolve_by_state",
    "resolve_feature_matrix_state",
    "sync_by_state_attributes",
    "build_penalty_definition",
    "build_selection_penalty_definition",
    "SplineTerm1D",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
    "GPSmoothTerm",
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "TensorANOVASplineTerm",
]
