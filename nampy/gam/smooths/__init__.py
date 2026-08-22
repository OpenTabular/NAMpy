from .categorical.fs import FSmoothInteractionTerm, SZSmoothInteractionTerm
from .categorical.re import RandomEffectTerm
from .registry import make_smooth_term, register_smooth
from .shape.scop import ShapeConstrainedPSplineTerm
from .smooth_base import (
    RUNTIME_TERM_INTERFACE_CHECKLIST,
    BaseSmoothTerm,
    ByState,
    apply_numeric_by,
    build_penalty_definition,
    by_values_from_new_data,
    column_as_float,
    column_as_object,
    columns_as_float_matrix,
    linear_functional_basis,
    linear_functional_by_state,
    resolve_by_state,
    sync_by_state_attributes,
)
from .tensor.te import TensorProductSplineTerm
from .tensor.ti import InteractionTensorProductSplineTerm
from .univariate.cr import CubicSplineTerm
from .univariate.ps import PSplineTerm1D
from .univariate.tp import ThinPlateSplineTerm

# mgcv-facing smooth aliases keep formulas/tests readable without reintroducing
# the old module-level compatibility facades.
te = TensorProductSplineTerm
ti = InteractionTensorProductSplineTerm

cr = cs = cc = CubicSplineTerm
ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm
fs = FSmoothInteractionTerm
sz = SZSmoothInteractionTerm
re = RandomEffectTerm
mpi = mpd = mdcv = mdcx = micv = micx = cv = cx = po = dpo = ipo = (
    ShapeConstrainedPSplineTerm
)

__all__ = [
    "register_smooth",
    "make_smooth_term",
    "BaseSmoothTerm",
    "ByState",
    "RUNTIME_TERM_INTERFACE_CHECKLIST",
    "apply_numeric_by",
    "by_values_from_new_data",
    "column_as_float",
    "column_as_object",
    "columns_as_float_matrix",
    "linear_functional_basis",
    "linear_functional_by_state",
    "resolve_by_state",
    "sync_by_state_attributes",
    "build_penalty_definition",
    "CubicSplineTerm",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "FSmoothInteractionTerm",
    "SZSmoothInteractionTerm",
    "RandomEffectTerm",
    "ShapeConstrainedPSplineTerm",
    "te",
    "ti",
    "cr",
    "cs",
    "cc",
    "ps",
    "tp",
    "ts",
    "fs",
    "sz",
    "re",
    "mpi",
    "mpd",
    "mdcv",
    "mdcx",
    "micv",
    "micx",
    "cv",
    "cx",
    "po",
    "dpo",
    "ipo",
]
