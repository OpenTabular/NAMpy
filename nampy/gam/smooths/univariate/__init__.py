from .bs import DerivativeBSplineTerm1D
from .cr import CubicSplineTerm
from .ds import DuchonSplineTerm
from .gp import GaussianProcessTerm
from .ps import PSplineTerm1D
from .tp import ThinPlateSplineTerm

bs = DerivativeBSplineTerm1D
cr = cs = cc = CubicSplineTerm
ds = DuchonSplineTerm
gp = GaussianProcessTerm
cp = ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm

__all__ = [
    "DerivativeBSplineTerm1D",
    "CubicSplineTerm",
    "DuchonSplineTerm",
    "GaussianProcessTerm",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
]
__all__ += ["bs", "cr", "cs", "cc", "cp", "ds", "gp", "ps", "tp", "ts"]
