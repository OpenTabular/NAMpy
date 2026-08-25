from .bs import DerivativeBSplineTerm1D
from .cr import CubicSplineTerm
from .ds import DuchonSplineTerm
from .gp import GaussianProcessTerm
from .ps import PSplineTerm1D
from .sos import SphericalSplineTerm
from .tp import ThinPlateSplineTerm

bs = DerivativeBSplineTerm1D
cr = cs = cc = CubicSplineTerm
ds = DuchonSplineTerm
gp = GaussianProcessTerm
sos = SphericalSplineTerm
cp = ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm

__all__ = [
    "DerivativeBSplineTerm1D",
    "CubicSplineTerm",
    "DuchonSplineTerm",
    "GaussianProcessTerm",
    "SphericalSplineTerm",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
]
__all__ += ["bs", "cr", "cs", "cc", "cp", "ds", "gp", "sos", "ps", "tp", "ts"]
