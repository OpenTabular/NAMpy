"""GAMLSS family implementations mirroring mgcv/R/gamlss.r."""

from ._base import GamlssFamily
from .bccg import BCCGLikelihood
from .betabinomial import BetaBinomialLocationPrecision
from .boxcox_tail import BCPELikelihood, BCTLikelihood
from .gammals import GammalsFamily, gammals
from .gaulss import GaulssFamily, gaulss
from .gevlss import GevlssFamily, gevlss
from .gumbls import GumblsFamily
from .inflated_beta import InflatedBeta
from .multinom import MultinomFamily, multinom
from .positive import LogNormalLocationScale, WeibullLocationShape
from .semicontinuous import ZeroAdjustedGamma
from .shash import ShashFamily
from .twlss import TwlssFamily, twlss
from .ziplss import ZiplssFamily

gumbls = GumblsFamily
ziplss = ZiplssFamily
shash = ShashFamily

__all__ = [
    "BCCGLikelihood",
    "BCTLikelihood", "BCPELikelihood",
    "InflatedBeta",
    "ZeroAdjustedGamma",
    "BetaBinomialLocationPrecision",
    "LogNormalLocationScale", "WeibullLocationShape",
    "GamlssFamily",
    "GaulssFamily",
    "gaulss",
    "GammalsFamily",
    "gammals",
    "GumblsFamily", "gumbls", "ZiplssFamily", "ziplss",
    "MultinomFamily", "multinom", "GevlssFamily", "gevlss",
    "ShashFamily", "shash", "TwlssFamily", "twlss",
]
