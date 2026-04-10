"""Shared link and variance function objects for GLM families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm as _norm

from .._mgcv_constants import LINK_ETA_EXP_CLIP


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
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.log(mu)

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.exp(np.clip(eta, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP))

    def mu_eta(self, eta):
        mu = self.inverse(eta)
        return np.clip(mu, self.eps, None)

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / _safe_pow(mu, 2, self.eps)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 / _safe_pow(mu, 3, self.eps)

    def d4(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -6.0 / _safe_pow(mu, 4, self.eps)


@dataclass(frozen=True)
class InverseLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 1.0 / mu

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / np.clip(eta, self.eps, None)

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return -1.0 / np.clip(eta**2, self.eps, None)

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 / _safe_pow(mu, 3, self.eps)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -6.0 / _safe_pow(mu, 4, self.eps)

    def d4(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 24.0 / _safe_pow(mu, 5, self.eps)


@dataclass(frozen=True)
class LogitLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.log(mu / (1.0 - mu))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (
            1.0 + np.exp(-np.clip(eta, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP))
        )

    def mu_eta(self, eta):
        mu = self.inverse(eta)
        return np.clip(mu * (1.0 - mu), self.eps, None)

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = _safe_pow(mu, 2, self.eps) * _safe_pow(1.0 - mu, 2, self.eps)
        return (2.0 * mu - 1.0) / denom

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = _safe_pow(mu, 3, self.eps) * _safe_pow(1.0 - mu, 3, self.eps)
        return 2.0 * (3.0 * mu**2 - 3.0 * mu + 1.0) / denom

    def d4(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = _safe_pow(mu, 4, self.eps) * _safe_pow(1.0 - mu, 4, self.eps)
        return 6.0 * (4.0 * mu**3 - 6.0 * mu**2 + 4.0 * mu - 1.0) / denom


@dataclass(frozen=True)
class ProbitLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return _norm.ppf(mu)

    def inverse(self, eta):
        return np.clip(
            _norm.cdf(np.asarray(eta, dtype=np.float64)), self.eps, 1.0 - self.eps
        )

    def mu_eta(self, eta):
        return np.clip(_norm.pdf(np.asarray(eta, dtype=np.float64)), self.eps, None)

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return eta / _safe_pow(phi, 2, self.eps)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return (1.0 + 2.0 * eta**2) / np.clip(phi**3, self.eps, None)

    def d4(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        eta = _norm.ppf(mu)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        return eta * (7.0 + 6.0 * eta**2) / np.clip(phi**4, self.eps, None)


@dataclass(frozen=True)
class CloglogLink(LinkFunction):
    eps: float

    def __call__(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.log(-np.log(1.0 - mu))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(np.clip(eta, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP))
        return np.clip(1.0 - np.exp(-lam), self.eps, 1.0 - self.eps)

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(np.clip(eta, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP))
        return np.clip(lam * np.exp(-lam), self.eps, None)

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        lam = np.clip(-np.log(1.0 - mu), self.eps, None)
        one_m_mu = np.clip(1.0 - mu, self.eps, None)
        return (lam - 1.0) / _safe_pow(one_m_mu * lam, 2, self.eps)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        lam = np.clip(-np.log(1.0 - mu), self.eps, None)
        one_m_mu = np.clip(1.0 - mu, self.eps, None)
        return (lam + 2.0 * (lam - 1.0) ** 2) / _safe_pow(one_m_mu * lam, 3, self.eps)

    def d4(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        lam = np.clip(-np.log(1.0 - mu), self.eps, None)
        one_m_mu = np.clip(1.0 - mu, self.eps, None)
        num = 6.0 * (lam - 1.0) ** 3 + (3.0 * lam - 1.0) * (
            lam + 2.0 * (lam - 1.0) ** 2
        )
        return num / _safe_pow(one_m_mu * lam, 4, self.eps)


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
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.clip(mu * (1.0 - mu), self.eps, None)

    def d1(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return 1.0 - 2.0 * mu

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return -2.0 * np.ones_like(mu)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class PoissonVariance(VarianceFunction):
    def __call__(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.clip(mu, self.eps, None)

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
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.clip(mu**2, self.eps, None)

    def d1(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 * mu

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 * np.ones_like(mu)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.zeros_like(mu)


@dataclass(frozen=True)
class NegativeBinomialVariance(VarianceFunction):
    family: Any

    def __call__(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        theta = float(self.family.theta)
        return np.clip(mu + (mu**2) / theta, self.eps, None)

    def d1(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        theta = float(self.family.theta)
        return 1.0 + 2.0 * mu / theta

    def d2(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        theta = float(self.family.theta)
        return (2.0 / theta) * np.ones_like(mu)

    def d3(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.zeros_like(mu)


LINK_REGISTRY: dict[str, type[LinkFunction]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "inverse": InverseLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
}

VARIANCE_REGISTRY: dict[str, type[VarianceFunction]] = {
    "constant": ConstantVariance,
    "binomial": BinomialVariance,
    "poisson": PoissonVariance,
    "gamma": GammaVariance,
}
