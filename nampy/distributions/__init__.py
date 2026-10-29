"""Shared probability predictions without statistical or neural training imports.

Normal, Gamma, Student-t, beta, negative-binomial, lognormal, Weibull and beta-binomial adapters
are supplied. Additional laws can
implement :class:`nampy.distributions.DistributionLaw` without changing the result
container or either fitting engine. BCCG probabilities and response moments
are available, with a statistical fixed/automatic-REML adapter; neural fitting
uses its normalized likelihood. BCT and BCPE expose probability operations and
existence-checked numerical moments; fitting belongs to the backend families.
"""

from .bccg import BCCGLaw
from .boxcox import BCPELaw, BCTLaw
from .contracts import DistributionLaw, DistributionSpec
from .laws import (
    BetaBinomialLaw,
    BetaLaw,
    GammaLaw,
    LogNormalLaw,
    NegativeBinomialLaw,
    NormalLaw,
    StudentTLaw,
    WeibullLaw,
)
from .mixtures import AtomMixtureLaw
from .observations import ObservationTarget, TruncatedLaw, observation_log_prob
from .prediction import DistributionPrediction

__all__ = [
    "DistributionLaw",
    "DistributionSpec",
    "BCTLaw",
    "BCPELaw",
    "BCCGLaw",
    "AtomMixtureLaw",
    "BetaBinomialLaw",
    "DistributionPrediction",
    "ObservationTarget",
    "TruncatedLaw",
    "observation_log_prob",
    "BetaLaw",
    "GammaLaw",
    "NegativeBinomialLaw",
    "NormalLaw",
    "StudentTLaw",
    "LogNormalLaw",
    "WeibullLaw",
]
