from .cr import CubicSplineTerm
from .gp import GPSmoothTerm
from .ps import PSplineTerm1D
from .tp import ThinPlateSplineTerm

cr = cs = cc = CubicSplineTerm
ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm
gp = GPSmoothTerm

__all__ = ["CubicSplineTerm", "PSplineTerm1D", "ThinPlateSplineTerm", "GPSmoothTerm"]
__all__ += ["cr", "cs", "cc", "ps", "tp", "ts", "gp"]
