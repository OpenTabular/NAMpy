from .cubic_regression import SplineTerm1D
from .gp import GPSmoothTerm
from .pspline import PSplineTerm1D
from .thin_plate import ThinPlateSplineTerm

__all__ = ["SplineTerm1D", "PSplineTerm1D", "ThinPlateSplineTerm", "GPSmoothTerm"]
