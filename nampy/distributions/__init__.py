"""Shared probability predictions without statistical or neural training imports.

Normal, Gamma, Student-t, beta, negative-binomial, lognormal, Weibull and beta-binomial adapters
are supplied. Additional laws can
implement :class:`nampy.distributions.DistributionLaw` without changing the result
container or either fitting engine. BCCG probabilities and response moments
are available, with a statistical fixed/automatic-REML adapter; neural fitting
uses its normalized likelihood. BCT and BCPE expose probability operations and
existence-checked numerical moments; fitting belongs to the backend families.
"""

from importlib import import_module

from .bccg import BCCGLaw
from .boxcox import BCPELaw, BCTLaw
from .contracts import DistributionFitStatus, DistributionLaw, DistributionSpec
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

_COMPARISON_EXPORTS = frozenset({
    "DistributionCandidate", "DistributionComparison", "DistributionComparisonRow",
    "compare_distributions",
})


def __getattr__(name):
    # Comparison consumes diagnostics. Load it only when requested so a caller
    # can import diagnostics first without cycling through probability exports.
    if name not in _COMPARISON_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".comparison", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "DistributionFitStatus", "DistributionLaw", "DistributionSpec",
    "BCTLaw",
    "BCPELaw",
    "BCCGLaw",
    "DistributionCandidate",
    "DistributionComparison",
    "DistributionComparisonRow",
    "compare_distributions",
    "AtomMixtureLaw",
    "BetaBinomialLaw",
    "DistributionPrediction",
    "ObservationTarget", "TruncatedLaw", "observation_log_prob",
    "BetaLaw",
    "GammaLaw",
    "NegativeBinomialLaw",
    "NormalLaw",
    "StudentTLaw",
    "LogNormalLaw", "WeibullLaw",
]
