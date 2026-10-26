"""Ordinary inverse-Gaussian GLM: stats::inverse.gaussian and mgcv fix.family.*."""

import numpy as np
from scipy.stats import invgauss

from .family_base import GLMFamily, JointOuterStrategy


class InverseGaussianFamily(GLMFamily):
    """Inverse Gaussian with ``1/mu^2``, inverse, log or identity link."""

    name = "inverse_gaussian"
    supports_pirls = True
    supports_gcv = True
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    known_scale = None
    max_derivative_order = 4
    _variance_key = "inverse_gaussian"
    # This kernel uses family.ls, i.e. the ordinary-GLM joint scale algebra.
    joint_outer_strategy = JointOuterStrategy.GAMMA_SCALE

    def __init__(self, link="1/mu^2", **kwargs):
        link = str(link).lower()
        if link not in {"1/mu^2", "inverse", "log", "identity"}:
            raise ValueError(
                "inverse_gaussian links are 1/mu^2, inverse, log, identity"
            )
        self._link_key = self.link_name = link
        self.canonical_link = link == "1/mu^2"
        super().__init__(**kwargs)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y <= 0):
            raise ValueError("inverse_gaussian requires strictly positive targets")
        return y

    def initialize_mu(self, y):
        return np.asarray(y, dtype=float).copy()

    def valid_mu(self, mu):
        # stats::inverse.gaussian deliberately leaves this check unrestricted.
        return True

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        valid = np.all(np.isfinite(eta))
        if self.link_name == "1/mu^2":
            valid = valid and np.all(eta > 0)
        elif self.link_name == "inverse":
            valid = valid and np.all(eta != 0)
        return bool(valid)

    def deviance_obs(self, y, mu, weights=None):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        return self._check_weights(y, weights) * (y - mu) ** 2 / (y * mu**2)

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def deviance_derivatives_mu(self, y, mu, order=1):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        if order == 1:
            return 2 / mu**2 - 2 * y / mu**3
        if order == 2:
            return 6 * y / mu**4 - 4 / mu**3
        if order == 3:
            return 12 / mu**4 - 24 * y / mu**5
        if order == 4:
            return 120 * y / mu**6 - 48 / mu**5
        raise ValueError("deviance derivative order must be 1 through 4")

    def working_weight_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        a, b, c = (self.inverse_link_derivatives(eta, i) for i in (1, 2, 3))
        if y is None:
            return 2 * a * b / mu**3 - 3 * a**3 / mu**4
        d1, d2, d3 = (self.deviance_derivatives_mu(y, mu, i) for i in (1, 2, 3))
        return (d3 * a**3 + 3 * d2 * a * b + d1 * c) / 2

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        a, b, c, d = (self.inverse_link_derivatives(eta, i) for i in (1, 2, 3, 4))
        if y is None:
            return (
                2 * (b**2 + a * c) / mu**3 - 15 * a**2 * b / mu**4 + 12 * a**4 / mu**5
            )
        d1, d2, d3, d4 = (self.deviance_derivatives_mu(y, mu, i) for i in (1, 2, 3, 4))
        return (
            d4 * a**4 + 6 * d3 * a**2 * b + 3 * d2 * b**2 + 4 * d2 * a * c + d1 * d
        ) / 2

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        pearson = np.sum(self._check_weights(y, weights) * (y - mu) ** 2 / mu**3)
        return float(pearson / (len(y) - (0 if edf is None else edf)))

    def loglik_obs(self, y, mu, scale=1.0):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        return -0.5 * (
            np.log(2 * np.pi * scale * y**3) + (y - mu) ** 2 / (scale * y * mu**2)
        )

    def aic(self, y, mu, *, edf=0.0, scale=1.0, weights=None, dev=None):
        del scale
        y = np.asarray(y, dtype=float)
        w = self._check_weights(y, weights)
        dev = self.deviance(y, mu, w) if dev is None else float(dev)
        return float(
            np.sum(w) * (1 + np.log(dev / np.sum(w) * 2 * np.pi))
            + 3 * np.sum(np.log(y) * w)
            + 2
            + 2 * edf
        )

    def ls(self, y, w, n=None, scale=1.0):
        del n
        y = np.asarray(y, dtype=float)
        w = self._check_weights(y, w)
        ii = w > 0
        nobs = np.sum(ii)
        return np.array(
            [
                -0.5 * np.sum(np.log(2 * np.pi * scale * y[ii] ** 3))
                + 0.5 * np.sum(np.log(w[ii])),
                -nobs / (2 * scale),
                nobs / (2 * scale * scale),
            ]
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        return float(self.ls(y, weights, n, scale)[0])

    def quantile_residual_bounds(self, y, mu, *, weights=None, scale=1.0):
        mu = np.asarray(mu, dtype=float)
        w = self._check_weights(mu, weights)
        w = w + (w == 0)
        cdf = invgauss.cdf(y, mu=mu * scale / w, scale=w / scale)
        return cdf, cdf
