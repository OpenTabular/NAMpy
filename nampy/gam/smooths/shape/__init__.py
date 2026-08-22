"""Shape-constrained runtime smooth terms."""

from .bivariate import BivariateShapePSplineTerm
from .scop import ShapeConstrainedPSplineTerm

__all__ = ["BivariateShapePSplineTerm", "ShapeConstrainedPSplineTerm"]
