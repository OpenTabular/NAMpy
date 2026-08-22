"""Single registry for LSS distribution families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

import numpy as np

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
    infer_kwargs: Callable | None = None

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
    DistributionFamily("normal", NormalDistribution, metric_profile="normal"),
    DistributionFamily("poisson", PoissonDistribution, metric_profile="poisson"),
    DistributionFamily("gamma", GammaDistribution, metric_profile="gamma"),
    DistributionFamily("beta", BetaDistribution, metric_profile="beta"),
    DistributionFamily(
        "dirichlet",
        DirichletDistribution,
        metric_profile="dirichlet",
        infer_kwargs=_infer_dirichlet,
    ),
    DistributionFamily("studentt", StudentTDistribution, metric_profile="studentt"),
    DistributionFamily("negativebinom", NegativeBinomialDistribution, metric_profile="negativebinom"),
    DistributionFamily("inversegamma", InverseGammaDistribution, metric_profile="inversegamma"),
    DistributionFamily(
        "categorical",
        CategoricalDistribution,
        metric_profile="categorical",
        infer_kwargs=_infer_categorical,
    ),
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
