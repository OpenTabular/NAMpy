"""Scaled Student t extended family, ported from mgcv/R/efam.r::scat."""

import warnings

import numpy as np
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import t as student_t

from ._function_maps import LINK_REGISTRY
from .family_base import ExtendedFamily, JointOuterStrategy


class ScaledTFamily(ExtendedFamily):
    """Location model with degrees of freedom and scale nuisance parameters."""

    pirls_curvature_mode = "observed"
    name = "scat"
    canonical_link = False
    supports_pirls = True
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.SCAT_THETA
    known_scale = 1.0
    max_derivative_order = 4

    def __init__(self, theta=None, link="identity", min_df=3.0):
        super().__init__()
        if link not in {"identity", "log", "inverse"}:
            raise ValueError("scat link must be 'identity', 'log', or 'inverse'.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)
        self.min_df = float(min_df)
        if not np.isfinite(self.min_df) or self.min_df < 0:
            raise ValueError("scat min_df must be finite and nonnegative.")
        self.n_theta = 2
        self.ini_theta = np.array([-2.0, -1.0])
        if theta is not None:
            theta = np.asarray(theta, dtype=float)
            if theta.shape != (2,) or not np.all(np.isfinite(theta)):
                raise ValueError("scat theta must contain two finite values.")
            if not np.any(theta == 0):
                if abs(theta[0]) <= self.min_df:
                    self.min_df = 0.9 * abs(theta[0])
                    warnings.warn(
                        "Supplied df below min.df. min.df reset", stacklevel=2
                    )
                self.ini_theta = np.log([abs(theta[0]) - self.min_df, abs(theta[1])])
                if not np.any(theta < 0):
                    self.n_theta = 0
        self._theta_working = self.ini_theta.copy()
        self._initialized_from_data = False

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        value = self._theta_working.copy()
        if trans:
            value = np.exp(value)
            value[0] += self.min_df
        return value

    def putTheta(self, theta):
        value = np.asarray(theta, dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("scat working theta must contain two finite values.")
        self._theta_working = value.copy()

    def _parameters(self, theta=None):
        theta = self.getTheta() if theta is None else np.asarray(theta, dtype=float)
        return np.exp(theta[0]) + self.min_df, np.exp(theta[1])

    def preinitialize(self, y):
        if self.estimate_theta and not self._initialized_from_data:
            theta = np.array([1.5, np.log(0.8 * np.std(np.asarray(y), ddof=1))])
            self.putTheta(theta)
            self._initialized_from_data = True
            return {"Theta": theta.copy()}
        return None

    def initialize_mu(self, y, weights=None):
        del weights
        y = self.validate_y(y)
        return y + (y == 0) * 0.1

    def initialize_linear_predictors(self, y):
        return [self.link(self.initialize_mu(y))]

    def valid_mu(self, mu):
        return bool(np.all(np.isfinite(mu)))

    def valid_eta(self, eta):
        if self.link_name != "inverse":
            return True  # stats::make.link identity/log valideta is unconditional.
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta != 0))

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def variance(self, mu):
        nu, sig = self._parameters()
        return np.full_like(np.asarray(mu, dtype=float), sig**2 * nu / (nu - 2))

    def dvar(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    d2var = dvar
    d3var = dvar

    @staticmethod
    def _weights(y, weights=None):
        return (
            np.ones_like(y, dtype=float)
            if weights is None
            else np.broadcast_to(np.asarray(weights, dtype=float), np.shape(y))
        )

    def deviance_obs(self, y, mu, weights=None, theta=None):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        nu, sig = self._parameters(theta)
        return (
            self._weights(y, weights) * (nu + 1) * np.log1p(((y - mu) / sig) ** 2 / nu)
        )

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        nu, sig = self._parameters()
        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - np.log(sig * np.sqrt(np.pi * nu))
            - self.deviance_obs(y, mu) / 2
        )

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale)))

    def aic(
        self, y, mu, theta=None, wt=None, dev=None, *, edf=0.0, scale=1.0, weights=None
    ):
        del dev, edf, scale
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        nu, sig = self._parameters(theta)
        constant = (
            -gammaln((nu + 1) / 2) + gammaln(nu / 2) + np.log(sig * np.sqrt(np.pi * nu))
        )
        w = self._weights(y, wt if weights is None else weights)
        return float(
            2 * np.sum(w * (constant + self.deviance_obs(y, mu, theta=theta) / 2))
        )

    def ls(self, y, w, theta=None, scale=1.0):
        del scale
        w = self._weights(y, w)
        nu, sig = self._parameters(theta)
        nu2 = nu - self.min_df
        nu2nu = nu2 / nu
        nu12 = (nu + 1) / 2
        value = gammaln(nu12) - gammaln(nu / 2) - np.log(sig * np.sqrt(np.pi * nu))
        grad = nu2 * digamma(nu12) / 2 - nu2 * digamma(nu / 2) / 2 - 0.5 * nu2nu
        individual = np.column_stack([w * grad, -w])
        term = (
            nu2**2 * polygamma(1, nu12) / 4
            + nu2 * digamma(nu12) / 2
            - nu2**2 * polygamma(1, nu / 2) / 4
            - nu2 * digamma(nu / 2) / 2
            + 0.5 * nu2nu**2
            - 0.5 * nu2nu
        )
        hessian = np.zeros((2, 2))
        hessian[0, 0] = np.sum(term * w)
        return {
            "ls": float(np.sum(value * w)),
            "lsth1": individual.sum(axis=0),
            "LSTH1": individual,
            "lsth2": hessian,
        }

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del n
        return self.ls(y, self._weights(y, weights), scale=scale)["ls"]

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        return 1.0

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        wt = self._weights(y, wt)
        nu, sig = self._parameters(theta)
        nu1 = nu + 1
        ym = y - mu
        nu2 = nu - self.min_df
        a = 1 + (ym / sig) ** 2 / nu
        nu1ym = nu1 * ym
        sig2a = sig**2 * a
        nusig2a = nu * sig2a
        f = nu1ym / nusig2a
        f1 = ym / nusig2a
        oo = {
            "Dmu": -2 * wt * f,
            "Dmu2": 2 * wt * nu1 * (1 / nusig2a - 2 * f1**2),
            "EDmu2": np.full(y.shape, 2 * nu1 / sig**2 / (nu + 3)),
        }
        if level > 0:
            nu1nusig2a = nu1 / nusig2a
            nu2nu = nu2 / nu
            fym, ff1, f1ym = f * ym, f * f1, f1 * ym
            fymf1 = fym * f1
            ymsig2a = ym / sig2a
            oo["Dth"] = np.column_stack(
                [wt * nu2 * (np.log(a) - fym / nu), -2 * wt * fym]
            )
            oo["Dmuth"] = np.column_stack(
                [2 * wt * (f - ymsig2a - fymf1) * nu2nu, 4 * wt * f * (1 - f1ym)]
            )
            oo["Dmu3"] = 4 * wt * f * (3 / nusig2a - 4 * f1**2)
            oo["Dmu2th"] = np.column_stack(
                [
                    2
                    * wt
                    * (
                        -nu1nusig2a
                        + 1 / sig2a
                        + 5 * ff1
                        - 2 * f1ym / sig2a
                        - 4 * fymf1 * f1
                    )
                    * nu2nu,
                    4 * wt * (-nu1nusig2a + ff1 * 5 - 4 * ff1 * f1ym),
                ]
            )
            oo["EDmu3"] = np.zeros_like(y)
            oo["EDmu2th"] = np.column_stack(
                [
                    np.full(
                        y.shape,
                        4
                        / (sig**2 * (nu + 3) ** 2)
                        * np.exp((self.getTheta() if theta is None else theta)[0]),
                    ),
                    -2 * oo["EDmu2"],
                ]
            )
        if level > 1:
            nu1nu = nu1 / nu
            fymf1ym, f1ymf1 = fym * f1ym, f1ym * f1
            oo["Dmu4"] = (
                12 * wt * (-nu1nusig2a / nusig2a + 8 * ff1 / nusig2a - 8 * ff1 * f1**2)
            )
            oo["Dmu3th"] = np.column_stack(
                [
                    4
                    * wt
                    * (
                        -6 * f / nusig2a
                        + 3 * f1 / sig2a
                        + 18 * ff1 * f1
                        - 4 * f1ymf1 / sig2a
                        - 12 * nu1ym * f1**4
                    )
                    * nu2nu,
                    48 * wt * f * (-1 / nusig2a + 3 * f1**2 - 2 * f1ymf1 * f1),
                ]
            )
            oo["Dth2"] = np.column_stack(
                [
                    wt
                    * (
                        nu2 * np.log(a)
                        + nu2nu
                        * ym**2
                        * (-2 * nu2 - nu1 + 2 * nu1 * nu2nu - nu1 * nu2nu * f1ym)
                        / nusig2a
                    ),
                    2 * wt * (fym - ym * ymsig2a - fymf1ym) * nu2nu,
                    4 * wt * fym * (1 - f1ym),
                ]
            )
            term = 2 * nu2nu - 2 * nu1nu * nu2nu - 1 + nu1nu
            oo["Dmuth2"] = np.column_stack(
                [
                    2
                    * wt
                    * f1
                    * nu2
                    * (
                        term
                        - 2 * nu2nu * f1ym
                        + 4 * fym * nu2nu / nu
                        - fym / nu
                        - 2 * fymf1ym * nu2nu / nu
                    ),
                    4
                    * wt
                    * (-f + ymsig2a + 3 * fymf1 - ymsig2a * f1ym - 2 * fymf1 * f1ym)
                    * nu2nu,
                    8 * wt * f * (-1 + 3 * f1ym - 2 * f1ym**2),
                ]
            )
            oo["Dmu2th2"] = np.column_stack(
                [
                    2
                    * wt
                    * nu2
                    * (
                        -term
                        + 10 * nu2nu * f1ym
                        - 16 * fym * nu2nu / nu
                        - 2 * f1ym
                        + 5 * nu1nu * f1ym
                        - 8 * nu2nu * f1ym**2
                        + 26 * fymf1ym * nu2nu / nu
                        - 4 * nu1nu * f1ym**2
                        - 12 * nu1nu * nu2nu * f1ym**3
                    )
                    / nusig2a,
                    4
                    * wt
                    * (
                        nu1nusig2a
                        - 1 / sig2a
                        - 11 * nu1 * f1**2
                        + 5 * f1ym / sig2a
                        + 22 * nu1 * f1ymf1 * f1
                        - 4 * f1ym**2 / sig2a
                        - 12 * nu1 * f1ymf1**2
                    )
                    * nu2nu,
                    8
                    * wt
                    * (
                        nu1nusig2a
                        - 11 * nu1 * f1**2
                        + 22 * nu1 * f1ymf1 * f1
                        - 12 * nu1 * f1ymf1**2
                    ),
                ]
            )
        return oo

    def deviance_derivatives_mu(self, y, mu, order=1):
        return self.Dd(y, mu, level=0 if order < 3 else order - 2)[
            "Dmu" if order == 1 else f"Dmu{order}"
        ]

    def score_eta(self, y, eta):
        return -0.5 * self.Dd(y, self.inverse_link(eta))["Dmu"] * self.mu_eta(eta)

    def hessian_eta(self, y, eta):
        dd = self.Dd(y, self.inverse_link(eta))
        return -0.5 * (
            dd["Dmu2"] * self.mu_eta(eta) ** 2
            + dd["Dmu"] * self.inverse_link_derivatives(eta, 2)
        )

    def rd(self, mu, wt=None, scale=1.0, *, rng=None):
        nu, sig = self._parameters()
        rng = np.random.default_rng() if rng is None else rng
        return rng.standard_t(nu, size=np.shape(mu)) * sig + np.asarray(mu)

    def quantile_residual_bounds(self, y, mu, *, weights=None, scale=1.0):
        nu, sig = self._parameters()
        value = student_t.cdf((np.asarray(y) - np.asarray(mu)) / sig, df=nu)
        return value, value
