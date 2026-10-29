"""Single registry for LSS distribution families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

import numpy as np

from ...distributions import (
    BCCGLaw,
    BCPELaw,
    BCTLaw,
    BetaBinomialLaw,
    BetaLaw,
    GammaLaw,
    LogNormalLaw,
    NegativeBinomialLaw,
    NormalLaw,
    StudentTLaw,
    WeibullLaw,
)
from ...distributions.mixtures import INFLATED_BETA_LAW, ZERO_ADJUSTED_GAMMA_LAW
from ..objectives.quantile import Quantile
from .boxcox_tail import BCPEDistribution, BCTDistribution
from .categorical import CategoricalDistribution, OrdinalCumulativeLogitDistribution
from .continuous import (
    BetaDistribution,
    GammaDistribution,
    InverseGammaDistribution,
    LogLogisticDistribution,
    LogNormalDistribution,
    NormalDistribution,
    RobustNormalDistribution,
    StudentTDistribution,
    WeibullDistribution,
)
from .counts import (
    HurdleNegativeBinomialDistribution,
    HurdlePoissonDistribution,
    NegativeBinomialDistribution,
    PoissonDistribution,
    TweedieDistribution,
    ZeroInflatedNegativeBinomialDistribution,
    ZeroInflatedPoissonDistribution,
)
from .distribution_adapter import (
    inflated_beta_adapter,
    negative_binomial_prediction,
    observation_prediction,
    zero_adjusted_gamma_adapter,
)
from .gamlss import (
    BCCGDistribution,
    BetaBinomialDistribution,
    InflatedBetaDistribution,
    ZeroAdjustedGammaDistribution,
)
from .multivariate import DirichletDistribution, MultivariateNormalDiagDistribution
from .observations import (
    LogNormalObservationsDistribution,
    NormalObservationsDistribution,
    WeibullObservationsDistribution,
)


@dataclass(frozen=True)
class DistributionFamily:
    name: str
    distribution: Type
    metric_profile: str | None = None
    infer_kwargs: Callable | None = None
    law_factory: Callable | None = None
    prediction_adapter: Callable | None = None

    @property
    def supports_distribution_prediction(self):
        return self.law_factory is not None or self.prediction_adapter is not None

    def instantiate(self, y, kwargs=None):
        resolved = dict(kwargs or {})
        if self.infer_kwargs is not None:
            resolved = self.infer_kwargs(np.asarray(y), resolved)
        return self.distribution(**resolved), resolved


def _infer_dirichlet(y, kwargs):
    if "n_dim" not in kwargs:
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                "Dirichlet family requires y with shape (n_samples, K), K>=2."
            )
        kwargs["n_dim"] = int(y.shape[1])
    return kwargs


def _infer_categorical(y, kwargs):
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = int(
            y.shape[1] if y.ndim == 2 and y.shape[1] > 1 else np.unique(y).size
        )
    return kwargs


def _infer_ordinal(y, kwargs):
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = int(np.unique(y).size)
    if int(kwargs["num_classes"]) < 2:
        raise ValueError("Ordinal family requires at least two classes.")
    return kwargs


def _infer_mvnormdiag(y, kwargs):
    if "n_dim" not in kwargs and "dim" not in kwargs:
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                "MultivariateNormalDiag family requires y with shape "
                "(n_samples, K), K>=2."
            )
        kwargs["n_dim"] = int(y.shape[1])
    return kwargs


_FAMILIES = (
    DistributionFamily("bct", BCTDistribution, law_factory=BCTLaw),
    DistributionFamily("bcpe", BCPEDistribution, law_factory=BCPELaw),
    DistributionFamily(
        "lognormal_observation", LogNormalObservationsDistribution,
        law_factory=lambda: LogNormalObservationsDistribution().law,
        prediction_adapter=observation_prediction,
    ),
    DistributionFamily(
        "weibull_observation", WeibullObservationsDistribution,
        law_factory=lambda: WeibullObservationsDistribution().law,
        prediction_adapter=observation_prediction,
    ),
    DistributionFamily(
        "normal_observation",
        NormalObservationsDistribution,
        law_factory=NormalLaw,
        prediction_adapter=observation_prediction,
    ),
    DistributionFamily("bccg", BCCGDistribution, law_factory=BCCGLaw),
    DistributionFamily(
        "betabinomial", BetaBinomialDistribution, law_factory=BetaBinomialLaw
    ),
    DistributionFamily(
        "beinf",
        InflatedBetaDistribution,
        law_factory=lambda: INFLATED_BETA_LAW,
        prediction_adapter=inflated_beta_adapter,
    ),
    DistributionFamily(
        "zaga",
        ZeroAdjustedGammaDistribution,
        law_factory=lambda: ZERO_ADJUSTED_GAMMA_LAW,
        prediction_adapter=zero_adjusted_gamma_adapter,
    ),
    DistributionFamily(
        "normal", NormalDistribution, metric_profile="normal", law_factory=NormalLaw
    ),
    DistributionFamily("poisson", PoissonDistribution, metric_profile="poisson"),
    DistributionFamily(
        "gamma", GammaDistribution, metric_profile="gamma", law_factory=GammaLaw
    ),
    DistributionFamily(
        "beta", BetaDistribution, metric_profile="beta", law_factory=BetaLaw
    ),
    DistributionFamily(
        "dirichlet",
        DirichletDistribution,
        metric_profile="dirichlet",
        infer_kwargs=_infer_dirichlet,
    ),
    DistributionFamily(
        "studentt",
        StudentTDistribution,
        metric_profile="studentt",
        law_factory=StudentTLaw,
    ),
    DistributionFamily(
        "negativebinom",
        NegativeBinomialDistribution,
        metric_profile="negativebinom",
        law_factory=NegativeBinomialLaw,
        prediction_adapter=negative_binomial_prediction,
    ),
    DistributionFamily(
        "inversegamma", InverseGammaDistribution, metric_profile="inversegamma"
    ),
    DistributionFamily(
        "categorical",
        CategoricalDistribution,
        metric_profile="categorical",
        infer_kwargs=_infer_categorical,
    ),
    DistributionFamily("quantile", Quantile, metric_profile="quantile"),
    DistributionFamily(
        "robustnormal", RobustNormalDistribution, metric_profile="normal"
    ),
    DistributionFamily("lognormal", LogNormalDistribution, law_factory=LogNormalLaw),
    DistributionFamily("weibull", WeibullDistribution, law_factory=WeibullLaw),
    DistributionFamily("loglogistic", LogLogisticDistribution),
    DistributionFamily("zip", ZeroInflatedPoissonDistribution),
    DistributionFamily("zinb", ZeroInflatedNegativeBinomialDistribution),
    DistributionFamily("hurdlepoisson", HurdlePoissonDistribution),
    DistributionFamily("hurdlenegativebinom", HurdleNegativeBinomialDistribution),
    DistributionFamily("tweedie", TweedieDistribution),
    DistributionFamily(
        "ordinal", OrdinalCumulativeLogitDistribution, infer_kwargs=_infer_ordinal
    ),
    DistributionFamily(
        "mvnormdiag",
        MultivariateNormalDiagDistribution,
        infer_kwargs=_infer_mvnormdiag,
    ),
)

FAMILY_REGISTRY = {family.name: family for family in _FAMILIES}


def resolve_family(name: str) -> DistributionFamily:
    try:
        return FAMILY_REGISTRY[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown distribution family {name!r}.") from exc


def family_spec_for_instance(instance):
    """Exact types only: subclasses may define different probability semantics."""
    return next(
        (
            spec
            for spec in FAMILY_REGISTRY.values()
            if type(instance) is spec.distribution
        ),
        None,
    )


def family_name_for_instance(instance) -> str:
    """Resolve a fitted distribution instance through the single registry."""
    for family in FAMILY_REGISTRY.values():
        if isinstance(instance, family.distribution):
            return family.name
    raise ValueError(f"Unregistered distribution class {type(instance).__name__!r}.")


__all__ = [
    "DistributionFamily",
    "FAMILY_REGISTRY",
    "resolve_family",
    "family_name_for_instance",
]
