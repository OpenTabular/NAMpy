from .categorical.fs import FSmoothInteractionTerm, SZSmoothInteractionTerm
from .categorical.mrf import MarkovRandomFieldTerm
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
from .tensor.t2 import AlternativeTensorProductSplineTerm
from .tensor.te import TensorProductSplineTerm
from .tensor.ti import InteractionTensorProductSplineTerm
from .univariate.ad import AdaptiveSmoothTerm
from .univariate.bs import DerivativeBSplineTerm1D
from .univariate.cr import CubicSplineTerm
from .univariate.ds import DuchonSplineTerm
from .univariate.gp import GaussianProcessTerm
from .univariate.ps import PSplineTerm1D
from .univariate.sos import SphericalSplineTerm
from .univariate.tp import ThinPlateSplineTerm

# mgcv-facing smooth aliases keep formulas/tests readable without reintroducing
# the old module-level compatibility facades.
te = TensorProductSplineTerm
ti = InteractionTensorProductSplineTerm
t2 = AlternativeTensorProductSplineTerm

bs = DerivativeBSplineTerm1D
ad = AdaptiveSmoothTerm
cr = cs = cc = CubicSplineTerm
ds = DuchonSplineTerm
gp = GaussianProcessTerm
sos = SphericalSplineTerm
mrf = MarkovRandomFieldTerm
cp = ps = PSplineTerm1D
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
    "DuchonSplineTerm",
    "GaussianProcessTerm",
    "SphericalSplineTerm",
    "MarkovRandomFieldTerm",
    "DerivativeBSplineTerm1D",
    "AdaptiveSmoothTerm",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "AlternativeTensorProductSplineTerm",
    "FSmoothInteractionTerm",
    "SZSmoothInteractionTerm",
    "RandomEffectTerm",
    "ShapeConstrainedPSplineTerm",
    "te",
    "ti",
    "t2",
    "ad",
    "bs",
    "cr",
    "cs",
    "cc",
    "ds",
    "gp",
    "sos",
    "mrf",
    "cp",
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
