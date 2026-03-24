"""Univariate smooth term implementations."""

from ..smooths.univariate.cubic_regression import SplineTerm1D
from ..smooths.univariate.gp import GPSmoothTerm
from ..smooths.univariate.pspline import PSplineTerm1D
from ..smooths.univariate.thin_plate import ThinPlateSplineTerm

__all__ = ["SplineTerm1D", "PSplineTerm1D", "ThinPlateSplineTerm", "GPSmoothTerm"]
