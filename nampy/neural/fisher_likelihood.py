"""Normalized Torch likelihoods and small output-coordinate Fisher metrics.

This adapter deliberately does not evaluate training loss/regularizers. It uses
exact family types so custom losses or subclasses cannot inherit a likelihood
claim accidentally. Family transforms remain owned by the training family.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import distributions as dist

from .families.categorical import CategoricalDistribution
from .families.continuous import (
    BetaDistribution,
    GammaDistribution,
    InverseGammaDistribution,
    LogNormalDistribution,
    NormalDistribution,
    RobustNormalDistribution,
    StudentTDistribution,
    WeibullDistribution,
)
from .families.counts import NegativeBinomialDistribution, PoissonDistribution
from .objectives.core import (
    BinaryObjective,
    DistributionObjective,
    MulticlassObjective,
    RegressionObjective,
)
from .objectives.parameterized import ParameterizedObjective

_FAMILIES = {
    NormalDistribution: "normal",
    LogNormalDistribution: "lognormal",
    GammaDistribution: "gamma",
    InverseGammaDistribution: "inversegamma",
    BetaDistribution: "beta",
    WeibullDistribution: "weibull",
    PoissonDistribution: "poisson",
    CategoricalDistribution: "categorical",
    NegativeBinomialDistribution: "negativebinomial",
    StudentTDistribution: "studentt",
}


@dataclass(frozen=True)
class FisherLikelihood:
    """One declared response experiment, independent of fitting machinery."""

    kind: str
    family: ParameterizedObjective | None = None
    regression_scale: float | None = None

    @classmethod
    def from_objective(cls, objective, config):
        if type(objective) is RegressionObjective and objective.uses_default_loss:
            if config.regression_scale is None:
                raise ValueError(
                    "MSE Fisher information requires explicit regression_scale."
                )
            return cls("regression", regression_scale=config.regression_scale)
        if type(objective) is BinaryObjective:
            return cls("binary")
        if type(objective) is MulticlassObjective:
            return cls("multiclass")
        if type(objective) is DistributionObjective:
            family = objective.family
            kind = _FAMILIES.get(type(family))
            if type(family) is RobustNormalDistribution and family.rob is None:
                kind = "normal"
            if kind is not None:
                if config.estimator == "exact" and kind in {
                    "negativebinomial",
                    "studentt",
                }:
                    raise NotImplementedError(
                        f"Exact Fisher information is unavailable for {kind}; use a sampled estimator."
                    )
                if config.estimator == "negative_hessian" and any(
                    callable(value)
                    for name, value in vars(family).items()
                    if name.endswith("_transform")
                ):
                    raise NotImplementedError(
                        "Hessian Fisher requires declared smooth family transforms."
                    )
                return cls(kind, family=family)
        raise NotImplementedError(
            f"{type(objective).__name__} does not declare a supported normalized Fisher likelihood. "
            "Custom, quantile, censored and survival objectives need their own sampling experiment."
        )

    @property
    def held_fixed(self):
        return (
            ()
            if self.regression_scale is None
            else (f"observation_scale={self.regression_scale!r}",)
        )

    def distribution(self, raw):
        if self.kind == "regression":
            assert self.regression_scale is not None
            return dist.Normal(raw, self.regression_scale)
        if self.kind == "binary":
            return dist.Bernoulli(logits=raw)
        if self.kind == "multiclass":
            return dist.Categorical(logits=raw)
        assert self.family is not None
        p = self.family(raw)
        if self.kind in {"normal", "lognormal"}:
            factory = dist.Normal if self.kind == "normal" else dist.LogNormal
            return factory(p[:, 0], p[:, 1])
        if self.kind in {"gamma", "inversegamma"}:
            gamma_factory = dist.Gamma if self.kind == "gamma" else dist.InverseGamma
            return gamma_factory(p[:, 0], p[:, 1])
        if self.kind == "beta":
            return dist.Beta(p[:, 0], p[:, 1])
        if self.kind == "weibull":
            return dist.Weibull(p[:, 0], p[:, 1])
        if self.kind == "poisson":
            return dist.Poisson(p[:, 0])
        if self.kind == "categorical":
            return dist.Categorical(probs=p)
        if self.kind == "studentt":
            return dist.StudentT(p[:, 0], p[:, 1], p[:, 2])
        mean, alpha = p.T
        size = 1 / alpha
        probability = mean / (mean + size)
        eps = self.family.eps
        if torch.any(
            (alpha < eps)
            | (mean + size < eps)
            | (probability <= eps)
            | (probability >= 1 - eps)
        ):
            raise NotImplementedError(
                "Fisher information is unavailable in the clamped negative-binomial regime."
            )
        return dist.NegativeBinomial(size, probs=probability)

    def log_prob(self, raw, target):
        values = self.distribution(raw).log_prob(target)
        return values.reshape(len(raw), -1).sum(dim=1)

    def coordinates_and_information(self, raw):
        """Return differentiable coordinates and expected information in them.

        All matrices are output-sized. Normal/Gamma/Beta coordinates are natural
        response parameters (not necessarily exponential-family parameters).
        """
        if self.kind == "regression":
            assert self.regression_scale is not None
            q = raw.reshape(-1)
            return q, torch.eye(
                q.numel(), device=q.device, dtype=q.dtype
            ) / self.regression_scale**2
        if self.kind == "binary":
            p = torch.sigmoid(raw.reshape(-1))
            return raw.reshape(-1), torch.diag(p * torch.sigmoid(-raw.reshape(-1)))
        if self.kind == "multiclass":
            p = torch.softmax(raw.reshape(-1), dim=0)
            return raw.reshape(-1), torch.diag(p) - p[:, None] * p[None, :]
        distribution = self.distribution(raw)
        if self.kind == "categorical":
            p = distribution.probs.reshape(-1)
            return p, torch.diag(1 / p) - torch.ones_like(p[:, None] * p[None, :])
        assert self.family is not None
        coordinates = self.family(raw).reshape(-1)
        q = coordinates.double()
        if self.kind in {"normal", "lognormal"}:
            information = torch.diag(
                torch.stack([1 / q[1].square(), 2 / q[1].square()])
            )
        elif self.kind in {"gamma", "inversegamma"}:
            shape, rate = q
            information = torch.stack(
                [
                    torch.stack([torch.polygamma(1, shape), -1 / rate]),
                    torch.stack([-1 / rate, shape / rate.square()]),
                ]
            )
        elif self.kind == "beta":
            alpha, beta = q
            common = torch.polygamma(1, alpha + beta)
            information = torch.stack(
                [
                    torch.stack([torch.polygamma(1, alpha) - common, -common]),
                    torch.stack([-common, torch.polygamma(1, beta) - common]),
                ]
            )
        elif self.kind == "weibull":
            scale, shape = q
            cross = (np.euler_gamma - 1) / scale
            information = torch.stack(
                [
                    torch.stack([shape.square() / scale.square(), cross]),
                    torch.stack(
                        [
                            cross,
                            (np.pi**2 / 6 + (1 - np.euler_gamma) ** 2) / shape.square(),
                        ]
                    ),
                ]
            )
        elif self.kind == "poisson":
            information = (1 / q).reshape(1, 1)
        else:
            raise NotImplementedError(
                f"Exact Fisher information is unavailable for {self.kind}."
            )
        return coordinates, information
