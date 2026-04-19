from .cubic_regression import SplineTerm1D
from .gp import GPSmoothTerm
from .pspline import PSplineTerm1D
from .thin_plate import ThinPlateSplineTerm

# Backward-compatible short constructor names.
cr = cs = cc = SplineTerm1D
ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm
gp = GPSmoothTerm

__all__ = ["SplineTerm1D", "PSplineTerm1D", "ThinPlateSplineTerm", "GPSmoothTerm"]
__all__ += ["cr", "cs", "cc", "ps", "tp", "ts", "gp"]
