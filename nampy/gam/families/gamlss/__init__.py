"""GAMLSS family implementations mirroring mgcv/R/gamlss.r."""

from ._base import GamlssFamily
from .gammals import GammalsFamily, gammals
from .gaulss import GaulssFamily, gaulss
from .gevlss import GevlssFamily, gevlss
from .gumbls import GumblsFamily
from .multinom import MultinomFamily, multinom
from .shash import ShashFamily
from .twlss import TwlssFamily, twlss
from .ziplss import ZiplssFamily

gumbls = GumblsFamily
ziplss = ZiplssFamily
shash = ShashFamily

__all__ = [
    "GamlssFamily",
    "GaulssFamily",
    "gaulss",
    "GammalsFamily",
    "gammals",
    "GumblsFamily", "gumbls", "ZiplssFamily", "ziplss",
    "MultinomFamily", "multinom", "GevlssFamily", "gevlss",
    "ShashFamily", "shash", "TwlssFamily", "twlss",
]
