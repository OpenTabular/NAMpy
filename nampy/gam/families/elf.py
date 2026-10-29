"""qgam ELF fixed-scale extended family for the existing PIRLS/REML engine.

Bandwidth and loss scale are calibration inputs. No new response probability
API, ordinary-GAM family alias or alternative fitting engine is registered.

References
----------
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). Fast
  Calibrated Additive Quantile Regression.
  https://doi.org/10.1080/01621459.2020.1725521
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). qgam:
  Bayesian Nonparametric Quantile Regression Modeling in R.
  https://doi.org/10.18637/jss.v100.i09
"""

import numpy as np
from scipy.special import betaln, digamma, expit, polygamma

from ._function_maps import LINK_REGISTRY
from .family_base import ExtendedFamily


class ELFFamily(ExtendedFamily):
    pirls_curvature_mode = "observed"
    name = "calibrated_elf"
    link_name = "identity"
    canonical_link = False
    supports_pirls = True
    supports_reml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    supported_smoothing_optimizers = frozenset({"outer_newton", "bfgs"})
    max_derivative_order = 4
    known_scale = 1.0
    n_theta = 0
    estimate_theta = False

    def __init__(self, *, level, bandwidth, loss_scale):
        super().__init__()
        if (
            not 0 < level < 1
            or not np.isfinite([bandwidth, loss_scale]).all()
            or min(bandwidth, loss_scale) <= 0
        ):
            raise ValueError(
                "ELF needs a quantile in (0,1), positive bandwidth and loss scale."
            )
        self.level, self.bandwidth, self.loss_scale = level, bandwidth, loss_scale
        self.link = LINK_REGISTRY["identity"](eps=self.eps)
        self.ini_theta = np.array([np.log(loss_scale)])
        self._theta_working = self.ini_theta.copy()

    def getTheta(self, trans=False):
        return np.exp(self._theta_working) if trans else self._theta_working.copy()

    def putTheta(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape != (1,) or not np.isfinite(theta).all():
            raise ValueError("ELF fixed log loss scale requires one finite value.")
        self._theta_working = theta.copy()
        self.loss_scale = float(np.exp(theta[0]))

    def initialize_mu(self, y, weights=None):
        return np.full_like(np.asarray(y, dtype=float), np.quantile(y, self.level))

    def initialize_linear_predictors(self, y):
        return [self.initialize_mu(y)]

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def valid_mu(self, mu):
        return bool(np.isfinite(mu).all())

    def valid_eta(self, eta):
        return bool(np.isfinite(eta).all())

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        return 1.0

    def _loss_term(self, y, mu):
        co, tau = self.bandwidth, self.level
        residual = np.asarray(y) - np.asarray(mu)
        return (
            (1 - tau) * co * np.log1p(-tau)
            + co * tau * np.log(tau)
            - (1 - tau) * residual
            + co * np.logaddexp(0, residual / co)
        )

    def deviance_obs(self, y, mu, weights=None, theta=None):
        weights = np.ones(len(y)) if weights is None else np.asarray(weights)
        sigma = (
            self.loss_scale
            if theta is None
            else np.exp(np.asarray(theta).reshape(-1)[0])
        )
        return 2 * weights * self._loss_term(y, mu) / sigma

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def loglik_obs(self, y, mu, scale=1.0):
        co, sigma, tau = self.bandwidth, self.loss_scale, self.level
        residual = np.asarray(y) - np.asarray(mu)
        return (
            ((1 - tau) * residual - co * np.logaddexp(0, residual / co)) / sigma
            - np.log(co)
            - betaln(co * (1 - tau) / sigma, co * tau / sigma)
        )

    def weighted_loglik(self, y, mu, weights=None, scale=1.0):
        value = self.loglik_obs(y, mu, scale)
        return float(np.sum(value if weights is None else np.asarray(weights) * value))

    def loglik(self, y, mu, scale=1.0):
        return self.weighted_loglik(y, mu, scale=scale)

    def aic(
        self, y, mu, theta=None, wt=None, dev=None, *, edf=0.0, scale=1.0, weights=None
    ):
        if wt is not None and weights is not None:
            raise TypeError("Supply weights once.")
        return -2 * self.weighted_loglik(
            y, mu, weights=wt if weights is None else weights
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        weights = np.ones(len(y)) if weights is None else np.asarray(weights)
        return self.ls(y, weights, scale=scale)["ls"]

    def Dd(self, y, mu, theta, wt, level=0):
        co, tau = self.bandwidth, self.level
        sigma = np.exp(np.asarray(theta).reshape(-1)[0])
        p = expit((np.asarray(y) - np.asarray(mu)) / co)
        variance = p * (1 - p)
        out = {
            "Dmu": -2 * np.asarray(wt) * (p - 1 + tau) / sigma,
            "Dmu2": 2 * np.asarray(wt) * variance / (co * sigma),
        }
        out["EDmu2"] = out["Dmu2"]
        if level > 0:
            out.update(
                Dth=-2 * np.asarray(wt) * self._loss_term(y, mu) / sigma,
                Dmuth=-out["Dmu"],
                Dmu2th=-out["Dmu2"],
                Dmu3=-2 * np.asarray(wt) * variance * (1 - 2 * p) / (sigma * co**2),
            )
        if level > 1:
            out.update(
                Dth2=-out["Dth"],
                Dmuth2=out["Dmu"],
                Dmu2th2=out["Dmu2"],
                Dmu4=2
                * np.asarray(wt)
                * variance
                * (1 - 6 * variance)
                / (sigma * co**3),
                Dmu3th=-out["Dmu3"],
            )
        return out

    def ls(self, y, w, theta=None, scale=1.0):
        co, tau = self.bandwidth, self.level
        sigma = (
            self.loss_scale
            if theta is None
            else np.exp(np.asarray(theta).reshape(-1)[0])
        )
        a, b = co * (1 - tau) / sigma, co * tau / sigma
        log0 = (
            co * ((1 - tau) * np.log1p(-tau) + tau * np.log(tau)) / sigma
            - np.log(co)
            - betaln(a, b)
        )
        first = (
            co
            / sigma
            * (
                -(1 - tau) * np.log1p(-tau)
                - tau * np.log(tau)
                + (1 - tau) * digamma(a)
                + tau * digamma(b)
                - digamma(a + b)
            )
        )
        second = -first - (co / sigma) ** 2 * (
            (1 - tau) ** 2 * polygamma(1, a)
            + tau * tau * polygamma(1, b)
            - polygamma(1, a + b)
        )
        return {
            "ls": float(np.sum(w) * log0),
            "lsth1": np.array([np.sum(w) * first]),
            "LSTH1": (np.asarray(w) * first)[:, None],
            "lsth2": np.array([[np.sum(w) * second]]),
        }
