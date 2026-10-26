"""Censored Poisson extended family, mgcv 1.9-4 R/efam.r::cpois."""

import numpy as np
from scipy.special import gammaln

from ._censored_location import CensoredLocationFamily, unpack_censored
from ._cpois_helpers import dppois, log_dpois, log_ppois
from ._function_maps import LINK_REGISTRY
from ._scalar_math import cube, exp
from .family_base import ExtendedFamily


class CensoredPoissonFamily(CensoredLocationFamily):
    """A Poisson mean predictor with exact, interval or one-sided censoring.

    Source interval probability is P(lo < Y <= hi). Source likelihood hooks
    deliberately ignore wt; observation weights are not frequency multipliers.
    """

    name = "cpois"
    canonical_link = False
    supports_pirls = True
    supports_gcv = False
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    known_scale = 1.0
    max_derivative_order = 4
    n_theta = 0

    def __init__(self, link="log"):
        super().__init__()
        if link not in {"identity", "log", "sqrt"}:
            raise ValueError("cpois link must be 'identity', 'log', or 'sqrt'.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)

    getTheta = ExtendedFamily.getTheta
    putTheta = ExtendedFamily.putTheta

    def valid_mu(self, mu):
        mu = np.asarray(mu, float)
        return bool(np.all(np.isfinite(mu)) if self.link_name == "identity" else
                    np.all(mu >= 0) if self.link_name == "sqrt" else np.all(mu > 0))

    @staticmethod
    def _masks(y):
        values, bounds = unpack_censored(y)
        exact = values == bounds
        return values, bounds, exact, np.isfinite(bounds) & ~exact

    def weighted_loglik_obs(self, y, mu, weights=None, scale=1.0):
        del weights, scale
        values, bounds, exact, interval = self._masks(y)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        result = np.zeros_like(values)
        result[exact] = log_dpois(values[exact], mu[exact])
        result[interval] = dppois(np.minimum(values[interval], bounds[interval]),
                                  np.maximum(values[interval], bounds[interval]), mu[interval])
        for lower, mask in ((True, np.isneginf(bounds)), (False, np.isposinf(bounds))):
            result[mask] = log_ppois(values[mask], mu[mask], lower)
        return result

    def weighted_loglik(self, y, mu, weights=None, scale=1.0):
        return float(np.sum(self.weighted_loglik_obs(y, mu, weights=weights, scale=scale)))

    def loglik_obs(self, y, mu, scale=1.0):
        return self.weighted_loglik_obs(y, mu, scale=scale)

    def loglik(self, y, mu, scale=1.0):
        return self.weighted_loglik(y, mu, scale=scale)

    def aic(self, y, mu, theta=None, wt=None, dev=None, *, weights=None):
        del theta, dev
        return -2 * self.weighted_loglik(y, mu, weights=weights if weights is not None else wt)

    def ls(self, y, w=None, theta=None, scale=1.0):
        del w, theta, scale
        values, bounds, exact, interval = self._masks(y)
        terms = np.zeros_like(values)
        terms[exact] = log_dpois(values[exact], values[exact])
        lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
        with np.errstate(all="ignore"):
            saturated_mu = exp((gammaln(np.floor(hi) + 1) - gammaln(np.floor(lo) + 1)) /
                               (np.floor(hi) - np.floor(lo)))
        terms[interval] = dppois(lo, hi, saturated_mu)
        return {"ls": float(np.sum(terms)), "lsth1": 0.0,
                "LSTH1": np.zeros((len(values), 1)), "lsth2": 0.0}

    def deviance_obs(self, y, mu, weights=None, theta=None):
        del weights, theta
        values, bounds, exact, interval = self._masks(y)
        terms = -2 * self.weighted_loglik_obs(y, mu)
        terms[exact] += 2 * log_dpois(values[exact], values[exact])
        lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
        with np.errstate(all="ignore"):
            saturated_mu = exp((gammaln(np.floor(hi) + 1) - gammaln(np.floor(lo) + 1)) /
                               (np.floor(hi) - np.floor(lo)))
        terms[interval] += 2 * dppois(lo, hi, saturated_mu)
        return terms

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        del theta, wt
        values, bounds, exact, interval = self._masks(y)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        ratios = []
        for order in range(1, 3 + int(level)):
            ratio = np.zeros_like(values)
            with np.errstate(all="ignore"):
                ratio[exact] = exp(log_dpois(values[exact] - order, mu[exact]) - log_dpois(values[exact], mu[exact]))
                lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
                ratio[interval] = exp(dppois(lo - order, hi - order, mu[interval]) - dppois(lo, hi, mu[interval]))
                for lower, mask in ((True, np.isneginf(bounds)), (False, np.isposinf(bounds))):
                    ratio[mask] = exp(log_ppois(values[mask] - order, mu[mask], lower) - log_ppois(values[mask], mu[mask], lower))
            ratios.append(ratio)
        f1, f2 = ratios[:2]
        result = {"Dmu": -2 * (f1 - 1), "Dmu2": -2 * (f2 - f1**2)}
        result["EDmu2"] = result["Dmu2"].copy()
        if level > 0:
            result["Dmu3"] = -2 * (ratios[2] - 3 * f1 * f2 + 2 * cube(f1))
        if level > 1:
            result["Dmu4"] = -2 * (ratios[3] - 4 * ratios[2] * f1 + 12 * f1**2 * f2 - 3 * f2**2 - 6 * f1**4)
        return result
