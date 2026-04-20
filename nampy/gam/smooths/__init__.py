from .categorical.factor_smooth import FSmoothInteractionTerm, SZSmoothInteractionTerm
from .categorical.mrf import MarkovRandomFieldTerm
from .categorical.random_effect import RandomEffectTerm
from .registry import available_smooths, make_smooth_term, register_smooth
from .smooth_base import (
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
from .tensor.t2 import TensorANOVASplineTerm
from .tensor.te import TensorProductSplineTerm
from .tensor.ti import InteractionTensorProductSplineTerm
from .univariate.cubic_regression import SplineTerm1D
from .univariate.gp import GPSmoothTerm
from .univariate.pspline import PSplineTerm1D
from .univariate.thin_plate import ThinPlateSplineTerm

# Backward-compatible short constructor names.
te = TensorProductSplineTerm
ti = InteractionTensorProductSplineTerm
t2 = TensorANOVASplineTerm

cc = cr = cs = SplineTerm1D
ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm
gp = GPSmoothTerm
fs = FSmoothInteractionTerm
sz = SZSmoothInteractionTerm
mrf = MarkovRandomFieldTerm
re = RandomEffectTerm

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
    "te",
    "ti",
    "t2",
    "cc",
    "cs",
    "cr",
    "ps",
    "tp",
    "ts",
    "gp",
    "fs",
    "sz",
    "mrf",
    "re",
]
