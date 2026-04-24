"""Shared link and variance function objects for GLM families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm


def _safe_pow(x, power, eps):
    x = np.asarray(x, dtype=np.float64)
    return np.clip(x**power, eps, None)


@dataclass(frozen=True)
class LinkFunction:
    def __call__(self, mu):
        raise NotImplementedError

    def inverse(self, eta):
        raise NotImplementedError

    def mu_eta(self, eta):
        raise NotImplementedError

    def d2(self, mu):
        raise NotImplementedError

    def d3(self, mu):
        raise NotImplementedError

    def d4(self, mu):
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        return np.asarray(mu, dtype=np.float64)

    def inverse(self, eta):
        return np.asarray(eta, dtype=np.float64)

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.ones_like(eta)

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class LogLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(mu)

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.exp(eta)

    def mu_eta(self, eta):
        mu = self.inverse(eta)
        return mu

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -1.0 / mu**2

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 2.0 / mu**3

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -6.0 / mu**4


@dataclass(frozen=True)
class InverseLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 1.0 / mu

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / eta

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return -1.0 / eta**2

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 2.0 / mu**3

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -6.0 / mu**4

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 24.0 / mu**5


@dataclass(frozen=True)
class SqrtLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.sqrt(mu)

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return eta**2

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 2.0 * eta

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -0.25 * mu**-1.5

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 0.375 * mu**-2.5

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -0.9375 * mu**-3.5


@dataclass(frozen=True)
class LogitLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(mu / (1.0 - mu))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return expit(eta)

    def mu_eta(self, eta):
        mu = self.inverse(eta)
        return mu * (1.0 - mu)

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 1.0 / (1.0 - mu) ** 2 - 1.0 / mu**2

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 2.0 / (1.0 - mu) ** 3 + 2.0 / mu**3

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 6.0 / (1.0 - mu) ** 4 - 6.0 / mu**4


@dataclass(frozen=True)
class ProbitLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return _norm.ppf(mu)

    def inverse(self, eta):
        return _norm.cdf(np.asarray(eta, dtype=np.float64))

    def mu_eta(self, eta):
        return _norm.pdf(np.asarray(eta, dtype=np.float64))

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return eta / _safe_pow(phi, 2, self.eps)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return (1.0 + 2.0 * eta**2) / np.clip(phi**3, self.eps, None)

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return eta * (7.0 + 6.0 * eta**2) / np.clip(phi**4, self.eps, None)


@dataclass(frozen=True)
class CloglogLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(-np.log(1.0 - mu))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(eta)
        return 1.0 - np.exp(-lam)

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(eta)
        return lam * np.exp(-lam)

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        l1m = np.log1p(-mu)
        return -1.0 / ((1.0 - mu) ** 2 * l1m) * (1.0 + 1.0 / l1m)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        l1m = np.log1p(-mu)
        return (-2.0 - 3.0 * l1m - 2.0 * l1m**2) / (
            (1.0 - mu) ** 3 * l1m**3
        )

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        l1m = np.log1p(-mu)
        return (-12.0 - 11.0 * l1m - 6.0 * l1m**2 - 6.0 / l1m) / (
            (1.0 - mu) ** 4 * l1m**3
        )


@dataclass(frozen=True)
class CauchitLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.tan(np.pi * (mu - 0.5))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 0.5 + np.arctan(eta) / np.pi

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (np.pi * (1.0 + eta**2))

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = np.tan(np.pi * (mu - 0.5))
        return 2.0 * np.pi**2 * eta * (1.0 + eta**2)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = np.tan(np.pi * (mu - 0.5))
        eta2 = eta**2
        return 2.0 * np.pi**3 * (1.0 + 3.0 * eta2) * (1.0 + eta2)

    def d4(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        eta = np.tan(np.pi * (mu - 0.5))
        eta2 = eta**2
        return 2.0 * np.pi**4 * (8.0 * eta + 12.0 * eta2 * eta) * (1.0 + eta2)


@dataclass(frozen=True)
class VarianceFunction:
    eps: float

    def __call__(self, mu):
        raise NotImplementedError

    def d1(self, mu):
        raise NotImplementedError

    def d2(self, mu):
        raise NotImplementedError

    def d3(self, mu):
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantVariance(VarianceFunction):
    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.ones_like(mu)

    def d1(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class BinomialVariance(VarianceFunction):
    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return mu * (1.0 - mu)

    def d1(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 1.0 - 2.0 * mu

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return -2.0 * np.ones_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class PoissonVariance(VarianceFunction):
    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return mu

    def d1(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.ones_like(mu)

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class GammaVariance(VarianceFunction):
    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return mu**2

    def d1(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 2.0 * mu

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return 2.0 * np.ones_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class NegativeBinomialVariance(VarianceFunction):
    family: Any

    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        theta = float(self.family.theta)
        return mu + (mu**2) / theta

    def d1(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        theta = float(self.family.theta)
        return 1.0 + 2.0 * mu / theta

    def d2(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        theta = float(self.family.theta)
        return (2.0 / theta) * np.ones_like(mu)

    def d3(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)


LINK_REGISTRY: dict[str, type[LinkFunction]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "inverse": InverseLink,
    "sqrt": SqrtLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
    "cauchit": CauchitLink,
}

VARIANCE_REGISTRY: dict[str, type[VarianceFunction]] = {
    "constant": ConstantVariance,
    "binomial": BinomialVariance,
    "poisson": PoissonVariance,
    "gamma": GammaVariance,
}
