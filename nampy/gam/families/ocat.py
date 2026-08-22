"""Ordered-categorical extended family, ported from ``mgcv::ocat``."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .._mgcv_constants import FAMILY_EPS
from ._function_maps import LINK_REGISTRY
from .family_base import ExtendedFamily, JointOuterStrategy


def _logistic_interval_probability(lower, upper):
    """Stable port of ocat's cancellation-resistant ``Fdiff`` helper."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    out = np.empty_like(lower)
    lower_positive = lower > 0.0
    upper_negative = upper < 0.0
    middle = ~(lower_positive | upper_negative)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if np.any(upper_negative):
            eb = np.exp(upper[upper_negative])
            ea = np.exp(lower[upper_negative])
            out[upper_negative] = eb / (1.0 + eb) - ea / (1.0 + ea)
        if np.any(lower_positive):
            eb = np.exp(-upper[lower_positive])
            ea = np.exp(-lower[lower_positive])
            out[lower_positive] = (ea - eb) / ((ea + 1.0) * (eb + 1.0))
        if np.any(middle):
            eb = np.exp(-upper[middle])
            ea = np.exp(lower[middle])
            out[middle] = (1.0 - ea * eb) / ((eb + 1.0) * (ea + 1.0))
    return out


def _abcd(x, level):
    """Port of the local ``ocat::Dd`` abcd helper."""
    x = np.asarray(x, dtype=np.float64)
    h = np.where(x > 0.0, -1.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        ex = np.exp(x * h)
        ex1 = ex + 1.0
        ex1_sq = ex1**2
        aj = -ex / ex1_sq
        if level < 0:
            return aj, None, None, None
        ex1_cube = ex1_sq * ex1
        ex2 = ex**2
        bj = h * (ex - ex2) / ex1_cube
        if level == 0:
            return aj, bj, None, None
        ex1_four = ex1_cube * ex1
        ex3 = ex2 * ex
        cj = (-ex3 + 4.0 * ex2 - ex) / ex1_four
        if level == 1:
            return aj, bj, cj, None
        ex1_five = ex1_four * ex1
        ex4 = ex3 * ex
        dj = h * (-ex4 + 11.0 * ex3 - 11.0 * ex2 + ex) / ex1_five
        return aj, bj, cj, dj


class OrderedCategoricalFamily(ExtendedFamily):
    """Ordered categorical family matching ``mgcv::ocat``."""

    name = "ocat"
    link_name = "identity"
    family_class = "extended"
    canonical_link = True

    supports_closed_form_solve = False
    supports_pirls = True
    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = False
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.OCAT_THETA
    use_fletcher_scale_estimate = False
    known_scale = 1.0
    max_derivative_order = 1

    def __init__(self, theta=None, R=None, link="identity", eps=FAMILY_EPS):
        super().__init__(eps=eps)
        if str(link).lower() != "identity":
            raise ValueError('ocat only supports the "identity" link.')
        if theta is None and R is None:
            raise ValueError("ocat requires theta or R.")
        if theta is not None:
            theta_arr = np.asarray(theta, dtype=np.float64).ravel()
            if theta_arr.size == 0:
                raise ValueError("ocat theta must contain at least one cutpoint gap.")
            R = int(theta_arr.size + 2)
        else:
            R = int(R)
            theta_arr = None
        if R < 3:
            raise ValueError("ocat requires R >= 3 categories.")

        self.R = R
        self.link = LINK_REGISTRY["identity"](eps=self.eps)
        if theta_arr is not None and np.all(theta_arr != 0.0):
            if np.any(theta_arr < 0.0):
                ini_theta = np.log(np.abs(theta_arr))
                self.n_theta = R - 2
            else:
                ini_theta = np.log(theta_arr)
                self.n_theta = 0
        else:
            ini_theta = np.full(R - 2, -1.0, dtype=np.float64)
            self.n_theta = R - 2
        self.ini_theta = np.asarray(ini_theta, dtype=np.float64).copy()
        self._theta_working = self.ini_theta.copy()
        self._initialized_from_data = False

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        theta = np.asarray(self._theta_working, dtype=np.float64).copy()
        if not trans:
            return theta
        alpha = np.empty(self.R - 1, dtype=np.float64)
        alpha[0] = -1.0
        if alpha.size > 1:
            alpha[1:] = -1.0 + np.cumsum(np.exp(theta))
        return alpha

    def putTheta(self, theta):
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if theta.shape != (self.R - 2,) or np.any(~np.isfinite(theta)):
            raise ValueError(f"ocat theta must have shape ({self.R - 2},).")
        self._theta_working = theta.copy()

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y != np.floor(y)) or np.any(y < 1.0) or np.any(y > self.R):
            raise ValueError(f"ocat responses must be integer labels in 1..{self.R}.")
        return y.astype(np.int64)

    def _initialize_theta_from_data(self, y):
        # Port of mgcv/R/efam.r::ocat.preinitialize::ocat.ini.
        y = np.concatenate((np.arange(1, self.R + 1), np.asarray(y)))
        counts = np.bincount(y.astype(np.int64), minlength=self.R + 1)[1:]
        p = np.cumsum(counts) / float(y.size)
        eta = 5.0 if p[0] == 0.0 else -1.0 - np.log(p[0] / (1.0 - p[0]))
        cut = np.full(self.R - 1, -1.0, dtype=np.float64)
        for i in range(1, self.R - 1):
            cut[i] = np.log(p[i] / (1.0 - p[i])) + eta
        gap = np.diff(cut)
        gap[gap <= 0.01] = 0.01
        self.putTheta(np.log(gap))

    def _initial_mu(self, y):
        if self.estimate_theta and not self._initialized_from_data:
            self._initialize_theta_from_data(y)
            self._initialized_from_data = True
        alpha = np.empty(self.R + 1, dtype=np.float64)
        alpha[0] = -2.0
        alpha[1] = -1.0
        if self.R > 2:
            alpha[2 : self.R] = -1.0 + np.cumsum(np.exp(self._theta_working))
        alpha[self.R] = alpha[self.R - 1] + 1.0
        y = np.asarray(y, dtype=np.int64)
        return 0.5 * (alpha[y] + alpha[y - 1])

    def initialize_mu(self, y, weights=None):
        del weights
        return self._initial_mu(self.validate_y(y))

    def initialize_linear_predictors(self, y):
        return [self.initialize_mu(y)]

    def inverse_link(self, eta):
        return np.asarray(eta, dtype=np.float64)

    def mu_eta(self, eta):
        return np.ones_like(np.asarray(eta, dtype=np.float64))

    def valid_mu(self, mu):
        return bool(np.all(np.isfinite(mu)))

    def valid_eta(self, eta):
        return bool(np.all(np.isfinite(eta)))

    def variance(self, mu):
        return np.ones_like(np.asarray(mu, dtype=np.float64))

    def dvar(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=np.float64))

    def d2var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=np.float64))

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=np.float64))

    def _alpha_for_deviance(self, theta=None):
        theta = self._theta_working if theta is None else theta
        theta = np.asarray(theta, dtype=np.float64).ravel()
        alpha = np.full(self.R + 1, np.inf, dtype=np.float64)
        alpha[0] = -np.inf
        alpha[1] = -1.0
        alpha[2 : self.R] = -1.0 + np.cumsum(np.exp(theta))
        return alpha

    def _probability(self, y, mu, theta=None):
        alpha = self._alpha_for_deviance(theta)
        y = np.asarray(y, dtype=np.int64)
        mu = np.asarray(mu, dtype=np.float64)
        return _logistic_interval_probability(alpha[y - 1] - mu, alpha[y] - mu)

    def deviance_obs(self, y, mu, weights=None, theta=None):
        y = self.validate_y(y)
        wt = (
            np.ones_like(y, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        probability = np.maximum(self._probability(y, mu, theta), np.finfo(float).tiny)
        return -2.0 * wt * np.log(probability)

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights=weights)))

    def _residual_cutpoints(self):
        return np.concatenate(([-np.inf], self.getTheta(trans=True), [np.inf]))

    def residuals(self, y, mu, rtype="deviance", eta=None, weights=None):
        """Mirror ``mgcv::ocat$residuals``."""
        y = self.validate_y(y)
        mu = np.asarray(mu, dtype=np.float64).ravel()
        eta = mu if eta is None else np.asarray(eta, dtype=np.float64).ravel()
        weights = (
            np.ones_like(mu, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).ravel()
        )
        rtype = str(rtype).lower()
        if rtype == "working":
            return y - eta
        if rtype == "response":
            category = np.searchsorted(self._residual_cutpoints(), eta, side="left")
            return y - category
        if rtype == "deviance":
            alpha = self._residual_cutpoints()
            sign = np.sign(0.5 * (alpha[y] + alpha[y - 1]) - eta)
            deviance = self.deviance_obs(y, mu, weights=weights)
            return sign * np.sqrt(np.maximum(deviance, 0.0))
        raise ValueError(
            "ocat residuals type must be one of "
            "{'deviance', 'working', 'response'}."
        )

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        probability = np.maximum(self._probability(y, mu), np.finfo(float).tiny)
        return np.log(probability)

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def aic(self, y, mu, theta=None, wt=None, dev=None, **kwargs):
        del dev, kwargs
        wt = np.ones_like(np.asarray(y, dtype=np.float64)) if wt is None else wt
        probability = np.maximum(self._probability(y, mu, theta), np.finfo(float).tiny)
        return float(-2.0 * np.sum(np.log(probability) * wt))

    def ls(self, y, w, theta, scale):
        del theta, scale
        y = np.asarray(y)
        w = np.ones_like(y, dtype=np.float64) if w is None else np.asarray(w)
        n = y.size
        return {
            "ls": 0.0,
            "lsth1": np.zeros(self.R - 2, dtype=np.float64),
            "LSTH1": np.zeros((n, self.R - 2), dtype=np.float64),
            "lsth2": np.zeros((self.R - 2, self.R - 2), dtype=np.float64),
        }

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        y = self.validate_y(y)
        mu = np.asarray(mu, dtype=np.float64)
        theta = self._theta_working if theta is None else np.asarray(theta)
        theta = np.asarray(theta, dtype=np.float64).ravel()
        wt = (
            np.ones_like(mu, dtype=np.float64)
            if wt is None
            else np.asarray(wt, dtype=np.float64)
        )
        alpha = self._alpha_for_deviance(theta)
        al0 = alpha[y - 1]
        al1 = alpha[y]
        x0 = al0 - mu
        x1 = al1 - mu
        f = np.maximum(_logistic_interval_probability(x0, x1), np.finfo(float).tiny)
        a1, b1, c1, d1 = _abcd(x1, level)
        a0, b0, c0, d0 = _abcd(x0, level)
        a = a1 - a0
        a2 = a * a
        out = {
            "D": -2.0 * wt * np.log(f),
            "Dmu": -2.0 * wt * a / f,
            "Dmu2": 2.0 * wt * (a2 / f - (b1 - b0)) / f,
        }
        out["EDmu2"] = out["Dmu2"].copy()
        if level <= 0:
            return out

        b = b1 - b0
        c = c1 - c0
        out["Dmu3"] = 2.0 * wt * (-c - 2.0 * a2 * a / f**2 + 3.0 * a * b / f) / f
        Dmua0 = 2.0 * (a0 * a / f - b0) / f
        Dmua1 = -2.0 * (a1 * a / f - b1) / f
        Dmu2a0 = -2.0 * (c0 + (a0 * (2.0 * a2 / f - b) - 2.0 * b0 * a) / f) / f
        Dmu2a1 = 2.0 * (c1 + (2.0 * (a1 * a2 / f - b1 * a) - a1 * b) / f) / f
        Da0 = -2.0 * a0 / f
        Da1 = 2.0 * a1 / f
        ntheta = self.R - 2
        out["Dth"] = np.zeros((y.size, ntheta), dtype=np.float64)
        out["Dmuth"] = np.zeros_like(out["Dth"])
        out["Dmu2th"] = np.zeros_like(out["Dth"])
        for k in range(ntheta):
            ek = np.exp(theta[k])
            ind = y == k + 2
            out["Dth"][ind, k] = wt[ind] * Da1[ind] * ek
            out["Dmuth"][ind, k] = wt[ind] * Dmua1[ind] * ek
            out["Dmu2th"][ind, k] = wt[ind] * Dmu2a1[ind] * ek
            ind = (y > k + 2) & (y < self.R)
            out["Dth"][ind, k] = wt[ind] * (Da1[ind] + Da0[ind]) * ek
            out["Dmuth"][ind, k] = wt[ind] * (Dmua1[ind] + Dmua0[ind]) * ek
            out["Dmu2th"][ind, k] = wt[ind] * (Dmu2a1[ind] + Dmu2a0[ind]) * ek
            ind = y == self.R
            out["Dth"][ind, k] = wt[ind] * Da0[ind] * ek
            out["Dmuth"][ind, k] = wt[ind] * Dmua0[ind] * ek
            out["Dmu2th"][ind, k] = wt[ind] * Dmu2a0[ind] * ek
        out["EDmu2th"] = out["Dmu2th"].copy()
        if level <= 1:
            return out

        d = d1 - d0
        f2 = f**2
        out["Dmu4"] = 2.0 * wt * ((3.0 * b**2 + 4.0 * a * c) / f + a2 * (6.0 * a2 / f - 12.0 * b) / f2 - d) / f
        Dmu3a0 = 2.0 * ((a0 * c + 3.0 * c0 * a + 3.0 * b0 * b) / f - d0 + 6.0 * a * (a0 * a2 / f - b0 * a - a0 * b) / f2) / f
        Dmu3a1 = 2.0 * (d1 - (a1 * c + 3.0 * (c1 * a + b1 * b)) / f + 6.0 * a * (b1 * a - a1 * a2 / f + a1 * b) / f2) / f
        Dmua0a0 = 2.0 * (c0 + (2.0 * a0 * (b0 - a0 * a / f) - b0 * a) / f) / f
        Dmua1a1 = 2.0 * ((b1 * a + 2.0 * a1 * (b1 - a1 * a / f)) / f - c1) / f
        Dmua0a1 = 2.0 * (a0 * (2.0 * a1 * a / f - b1) - b0 * a1) / f2
        Dmu2a0a0 = 2.0 * (d0 + (b0 * (2.0 * b0 - b) + 2.0 * c0 * (a0 - a)) / f + 2.0 * (b0 * a2 + a0 * (3.0 * a0 * a2 / f - 4.0 * b0 * a - a0 * b)) / f2) / f
        Dmu2a1a1 = 2.0 * ((2.0 * c1 * (a + a1) + b1 * (2.0 * b1 + b)) / f + 2.0 * (a1 * (3.0 * a1 * a2 / f - a1 * b) - b1 * a * (a + 4.0 * a1)) / f2 - d1) / f
        Dmu2a0a1 = np.zeros_like(a)
        Da0a0 = 2.0 * (b0 + a0**2 / f) / f
        Da1a1 = -2.0 * (b1 - a1**2 / f) / f
        Da0a1 = -2.0 * a0 * a1 / f2
        pairs = [(j, k) for j in range(ntheta) for k in range(j, ntheta)]
        out["Dth2"] = np.zeros((y.size, len(pairs)), dtype=np.float64)
        out["Dmuth2"] = np.zeros_like(out["Dth2"])
        out["Dmu2th2"] = np.zeros_like(out["Dth2"])
        out["Dmu3th"] = np.zeros_like(out["Dth"])
        for pair_index, (j, k) in enumerate(pairs):
            ek = np.full(y.size, np.exp(theta[k]))
            ek1 = ek.copy()
            ek[(y == self.R) | (y <= k + 1)] = 0.0
            ek1[y < k + 3] = 0.0
            ej = np.full(y.size, np.exp(theta[j]))
            ej1 = ej.copy()
            ej[(y == self.R) | (y <= j + 1)] = 0.0
            ej1[y < j + 3] = 0.0
            ekj = np.zeros(y.size)
            ekj1 = np.zeros(y.size)
            if j == k:
                ekj[(y > k + 1) & (y < self.R)] = np.exp(theta[k])
                ekj1[y > k + 2] = np.exp(theta[k])
                ind = y >= j + 1
                out["Dmu3th"][ind, k] = wt[ind] * (Dmu3a1[ind] * ek[ind] + Dmu3a0[ind] * ek1[ind])
            out["Dth2"][:, pair_index] = wt * (
                Da1a1 * ek * ej
                + Da0a1 * ek * ej1
                + Da1 * ekj
                + Da0a0 * ek1 * ej1
                + Da0a1 * ek1 * ej
                + Da0 * ekj1
            )
            out["Dmuth2"][:, pair_index] = wt * (
                Dmua1a1 * ek * ej
                + Dmua0a1 * ek * ej1
                + Dmua1 * ekj
                + Dmua0a0 * ek1 * ej1
                + Dmua0a1 * ek1 * ej
                + Dmua0 * ekj1
            )
            out["Dmu2th2"][:, pair_index] = wt * (
                Dmu2a1a1 * ek * ej
                + Dmu2a0a1 * ek * ej1
                + Dmu2a1 * ekj
                + Dmu2a0a0 * ek1 * ej1
                + Dmu2a0a1 * ek1 * ej
                + Dmu2a0 * ekj1
            )
        return out

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        del y, mu, edf, weights
        return 1.0

    def _response_probabilities_and_derivative(self, eta):
        eta = np.asarray(eta, dtype=np.float64).ravel()
        alpha = self.getTheta(trans=True)
        cdf = np.empty((eta.size, self.R + 1), dtype=np.float64)
        dcdf = np.zeros_like(cdf)
        cdf[:, 0] = 0.0
        for i, cut in enumerate(alpha, start=1):
            probability = expit(cut - eta)
            cdf[:, i] = probability
            dcdf[:, i] = probability * (probability - 1.0)
        cdf[:, -1] = 1.0
        return np.diff(cdf, axis=1), np.diff(dcdf, axis=1)

    def response_from_eta(self, eta):
        probabilities, _derivative = self._response_probabilities_and_derivative(eta)
        return probabilities

    def response_se_from_eta(self, eta, eta_se):
        _probabilities, derivative = self._response_probabilities_and_derivative(eta)
        eta_se = np.asarray(eta_se, dtype=np.float64).ravel()
        return np.abs(derivative) * eta_se[:, None]
