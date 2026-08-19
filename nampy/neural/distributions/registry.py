"""Single registry for LSS distribution families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from .distributions import (
    BetaDistribution,
    CategoricalDistribution,
    DirichletDistribution,
    GammaDistribution,
    HurdleNegativeBinomialDistribution,
    HurdlePoissonDistribution,
    InverseGammaDistribution,
    LogLogisticDistribution,
    LogNormalDistribution,
    MultivariateNormalDiagDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    OrdinalCumulativeLogitDistribution,
    PoissonDistribution,
    Quantile,
    RobustNormalDistribution,
    StudentTDistribution,
    TweedieDistribution,
    WeibullDistribution,
    ZeroInflatedNegativeBinomialDistribution,
    ZeroInflatedPoissonDistribution,
)


@dataclass(frozen=True)
class DistributionFamily:
    name: str
    distribution: Type
    metric_profile: str | None = None


_FAMILIES = (
    DistributionFamily("normal", NormalDistribution, metric_profile="normal"),
    DistributionFamily("poisson", PoissonDistribution, metric_profile="poisson"),
    DistributionFamily("gamma", GammaDistribution, metric_profile="gamma"),
    DistributionFamily("beta", BetaDistribution, metric_profile="beta"),
    DistributionFamily("dirichlet", DirichletDistribution, metric_profile="dirichlet"),
    DistributionFamily("studentt", StudentTDistribution, metric_profile="studentt"),
    DistributionFamily("negativebinom", NegativeBinomialDistribution, metric_profile="negativebinom"),
    DistributionFamily("inversegamma", InverseGammaDistribution, metric_profile="inversegamma"),
    DistributionFamily("categorical", CategoricalDistribution, metric_profile="categorical"),
    DistributionFamily("quantile", Quantile, metric_profile="quantile"),
    DistributionFamily("robustnormal", RobustNormalDistribution, metric_profile="normal"),
    DistributionFamily("lognormal", LogNormalDistribution),
    DistributionFamily("weibull", WeibullDistribution),
    DistributionFamily("loglogistic", LogLogisticDistribution),
    DistributionFamily("zip", ZeroInflatedPoissonDistribution),
    DistributionFamily("zinb", ZeroInflatedNegativeBinomialDistribution),
    DistributionFamily("hurdlepoisson", HurdlePoissonDistribution),
    DistributionFamily("hurdlenegativebinom", HurdleNegativeBinomialDistribution),
    DistributionFamily("tweedie", TweedieDistribution),
    DistributionFamily("ordinal", OrdinalCumulativeLogitDistribution),
    DistributionFamily("mvnormdiag", MultivariateNormalDiagDistribution),
)

FAMILY_REGISTRY = {family.name: family for family in _FAMILIES}


def resolve_family(name: str) -> DistributionFamily:
    try:
        return FAMILY_REGISTRY[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown distribution family {name!r}.") from exc


def family_name_for_instance(instance) -> str:
    """Resolve a fitted distribution instance through the single registry."""
    for family in _FAMILIES:
        if isinstance(instance, family.distribution):
            return family.name
    raise ValueError(f"Unregistered distribution class {type(instance).__name__!r}.")


__all__ = [
    "DistributionFamily", "FAMILY_REGISTRY", "resolve_family",
    "family_name_for_instance",
]
