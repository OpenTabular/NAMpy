from ..smooths.categorical.factor_smooth import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
)
from ..smooths.categorical.mrf import MarkovRandomFieldTerm
from ..smooths.categorical.random_effect import RandomEffectTerm
from ..smooths.tensor.t2 import TensorANOVASplineTerm
from ..smooths.tensor.te import TensorProductSplineTerm
from ..smooths.tensor.ti import InteractionTensorProductSplineTerm
from ..smooths.univariate.cubic_regression import SplineTerm1D
from ..smooths.univariate.gp import GPSmoothTerm
from ..smooths.univariate.pspline import PSplineTerm1D
from ..smooths.univariate.thin_plate import ThinPlateSplineTerm
from .linear import LinearTerm

__all__ = [
    "LinearTerm",
    "SplineTerm1D",
    "PSplineTerm1D",
    "ThinPlateSplineTerm",
    "GPSmoothTerm",
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "TensorANOVASplineTerm",
    "RandomEffectTerm",
    "FSmoothInteractionTerm",
    "SZSmoothInteractionTerm",
    "MarkovRandomFieldTerm",
]
