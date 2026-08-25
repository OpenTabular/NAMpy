"""Beta-regression extended family, ported from ``mgcv::betar``."""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from ._function_maps import LINK_REGISTRY, BetaVariance
from .family_base import ExtendedFamily, JointOuterStrategy

_DEFAULT_BETAR_EPS = float(np.finfo(np.float64).eps * 100.0)


def _beta_log_density(y, mu, theta):
    alpha = theta * mu
    beta = theta * (1.0 - mu)
    return (
        gammaln(theta)
        - gammaln(alpha)
        - gammaln(beta)
        + (alpha - 1.0) * np.log(y)
        + (beta - 1.0) * np.log1p(-y)
    )


def _beta_saturated_loglik(y, weights, theta, eps):
    """Port of ``mgcv::betar()$saturated.ll`` Newton search."""
    y = np.asarray(y, dtype=np.float64).ravel().copy()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if y.shape != weights.shape:
        raise ValueError("y and weights must have matching shapes.")

    y[y < eps] = eps
    y[y > 1.0 - eps] = 1.0 - eps
    a = eps
    b = 1.0 - eps
    eta = np.asarray(y, dtype=np.float64).copy()
    eta[y <= eps * 1.2] = eps * 1.2
    eta[y >= 1.0 - eps * 1.2] = 1.0 - eps * 1.2
    eta = np.log((eta - a) / (b - eta))

    ls_store = np.ones_like(y)
    mu_store = np.ones_like(y)
    active = np.arange(y.size, dtype=np.int64)

    def gbh(yv, etav, deriv, local_a=1e-8):
        local_b = 1.0 - local_a
        positive = etav > 0.0
        expeta = np.empty_like(etav)
        expeta[positive] = np.exp(-etav[positive])
        expeta[~positive] = np.exp(etav[~positive])
        mu = np.empty_like(etav)
        mu[positive] = (local_a * expeta[positive] + local_b) / (1.0 + expeta[positive])
        mu[~positive] = (local_a + local_b * expeta[~positive]) / (
            1.0 + expeta[~positive]
        )
        log_density = _beta_log_density(yv, mu, theta)
        if not deriv:
            return log_density, None, None, mu

        grad_mu = theta * (
            np.log(yv)
            - np.log1p(-yv)
            - digamma(mu * theta)
            + digamma((1.0 - mu) * theta)
        )
        hess_mu = -(theta**2) * (
            polygamma(1, mu * theta) + polygamma(1, (1.0 - mu) * theta)
        )
        dmu_deta = expeta * (local_b - local_a) / (1.0 + expeta) ** 2
        d2mu_deta2 = (
            np.sign(etav)
            * ((local_a - local_b) * expeta + (local_b - local_a) * expeta**2)
            / (1.0 + expeta) ** 3
        )
        hess_eta = hess_mu * dmu_deta**2 + grad_mu * d2mu_deta2
        return log_density, grad_mu * dmu_deta, hess_eta, mu

    for _ in range(200):
        log_density, grad, hess, mu = gbh(y, eta, True, local_a=eps / 10.0)
        converged = np.abs(grad) < np.mean(np.abs(log_density) + 0.1) * 1e-8
        if np.any(converged):
            ls_store[active[converged]] = log_density[converged]
            mu_store[active[converged]] = mu[converged]
            active = active[~converged]
            if active.size == 0:
                break
            y = y[~converged]
            eta = eta[~converged]
            log_density = log_density[~converged]
            grad = grad[~converged]
            hess = hess[~converged]

        h = -hess
        hmin = float(np.max(h)) * 1e-4
        h = np.maximum(h, hmin)
        delta = grad / h
        delta = np.clip(delta, -2.0, 2.0)
        trial_log_density, _, _, _ = gbh(y, eta + delta, False, local_a=eps / 10.0)
        failed = trial_log_density < log_density
        for _ in range(20):
            if not np.any(failed):
                break
            delta[failed] *= 0.5
            trial_log_density[failed], _, _, _ = gbh(
                y[failed], eta[failed] + delta[failed], False, local_a=eps / 10.0
            )
            failed = trial_log_density < log_density
        eta += delta
    else:
        if active.size:
            ls_store[active] = log_density
            mu_store[active] = mu

    return float(np.sum(weights * ls_store)), ls_store, mu_store


class BetaRegressionFamily(ExtendedFamily):
    """``mgcv::betar`` for responses in ``(0, 1)``."""

    name = "betar"
    link_name = "logit"
    family_class = "extended"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True
    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.BETAR_THETA

    known_scale = 1.0
    max_derivative_order = 1

    def __init__(
        self,
        theta=None,
        link="logit",
        eps=_DEFAULT_BETAR_EPS,
    ):
        super().__init__(eps=float(eps))
        link_key = str(link).lower()
        if link_key not in {"logit", "probit", "cloglog", "cauchit"}:
            raise ValueError(
                "betar link must be one of 'logit', 'probit', 'cloglog', or 'cauchit'."
            )
        self.link_name = link_key
        self.link = LINK_REGISTRY[link_key](eps=self.eps)
        self.n_theta = 1
        if theta is not None and float(theta) != 0.0:
            theta = float(theta)
            if theta > 0.0:
                self.n_theta = 0
                self.ini_theta = float(np.log(theta))
            else:
                self.ini_theta = float(np.log(-theta))
        else:
            self.ini_theta = 0.0
        self._theta_working = float(self.ini_theta)
        self.variance = BetaVariance(eps=self.eps, family=self)

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        value = float(self._theta_working)
        return float(np.exp(value)) if trans else value

    def putTheta(self, theta):
        theta = float(theta)
        if not np.isfinite(theta):
            raise ValueError("betar requires a finite log(theta).")
        self._theta_working = theta

    def _theta_value(self, theta=None):
        value = self._theta_working if theta is None else float(theta)
        return float(np.exp(value))

    def _check_weights(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            return np.ones_like(y, dtype=np.float64)
        return np.asarray(weights, dtype=np.float64)

    def validate_y(self, y):
        y = super().validate_y(y)
        return np.clip(y, self.eps, 1.0 - self.eps)

    def initialize_mu(self, y, weights=None):
        del weights
        return self.validate_y(y)

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all((mu > 0.0) & (mu < 1.0)))

    def valid_eta(self, eta):
        return bool(np.all(np.isfinite(np.asarray(eta, dtype=np.float64))))

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def dvar(self, mu):
        return self.variance.d1(mu)

    def d2var(self, mu):
        return self.variance.d2(mu)

    def d3var(self, mu):
        return self.variance.d3(mu)

    def deviance_obs(self, y, mu, weights=None, theta=None):
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, 1.0 - self.eps)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        wt = self._check_weights(y, weights)
        theta_value = self._theta_value(theta)
        muth = mu * theta_value
        return (
            2.0
            * wt
            * (
                -gammaln(theta_value)
                + gammaln(muth)
                + gammaln(theta_value - muth)
                - muth * (np.log(y) - np.log1p(-y))
                - theta_value * np.log1p(-y)
                + np.log(y)
                + np.log1p(-y)
            )
        )

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights=weights)))

    def postprocess_deviance(self, y, mu, weights=None):
        """Mirror ``mgcv::betar()$postproc`` for reported deviance."""
        raw = self.deviance(y, mu, weights=weights)
        saturated = self.saturated_loglik(y, weights=weights)
        return float(raw + 2.0 * saturated)

    def null_deviance(self, y, mu, weights=None, offset=None, intercept=True):
        """Mirror the betar-specific null-deviance postprocessing."""
        y = np.asarray(y, dtype=np.float64).ravel()
        weights = self._check_weights(y, weights)
        if intercept:
            null_mu = np.full_like(y, np.sum(weights * y) / np.sum(weights))
        else:
            offset = np.zeros_like(y) if offset is None else np.asarray(offset)
            null_mu = self.inverse_link(offset)
        saturated = self.saturated_loglik(y, weights=weights)
        return float(2.0 * saturated + self.deviance(y, null_mu, weights=weights))

    def residuals(self, y, mu, rtype="deviance", eta=None, weights=None):
        """Mirror ``mgcv::betar()$residuals``."""
        del eta
        y = np.asarray(y, dtype=np.float64).ravel()
        mu = np.asarray(mu, dtype=np.float64).ravel()
        weights = self._check_weights(y, weights)
        rtype = str(rtype).lower()
        if rtype == "response":
            return y - mu
        if rtype == "pearson":
            return (y - mu) / np.sqrt(self.variance(mu))
        if rtype == "deviance":
            _, saturated_terms, _ = _beta_saturated_loglik(
                y, weights, self.getTheta(trans=True), self.eps
            )
            deviance = 2.0 * saturated_terms + self.deviance_obs(y, mu, weights)
            return np.sign(y - mu) * np.sqrt(np.maximum(deviance, 0.0))
        if rtype == "working":
            return y - mu
        raise ValueError(
            "betar residuals type must be one of "
            "{'deviance', 'working', 'response', 'pearson'}."
        )

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, 1.0 - self.eps)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return _beta_log_density(y, mu, self.getTheta(trans=True))

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def aic(
        self,
        y,
        mu,
        theta=None,
        wt=None,
        dev=None,
        *,
        edf=0.0,
        scale=1.0,
        weights=None,
    ):
        del dev, edf, scale
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        wt = self._check_weights(y, weights if weights is not None else wt)
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, 1.0 - self.eps)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        theta_value = self._theta_value(theta)
        return float(-2.0 * np.sum(wt * _beta_log_density(y, mu, theta_value)))

    def ls(self, y, w, theta=None, scale=1.0):
        del w, theta, scale
        return {
            "ls": 0.0,
            "lsth1": 0.0,
            "LSTH1": np.zeros((np.asarray(y).size, 1), dtype=np.float64),
            "lsth2": 0.0,
        }

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del n, scale
        theta_value = self.getTheta(trans=True)
        return _beta_saturated_loglik(
            y,
            self._check_weights(y, weights),
            theta_value,
            self.eps,
        )[0]

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        del y, mu, edf, weights
        return 1.0

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        """Port of ``mgcv/R/efam.r::betar()$Dd``."""
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, 1.0 - self.eps)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        wt = self._check_weights(y, wt)
        theta_value = self._theta_value(theta)
        onemu = 1.0 - mu
        muth = mu * theta_value
        onemuth = onemu * theta_value
        psi0_th = digamma(theta_value)
        psi1_th = polygamma(1, theta_value)
        psi0_muth = digamma(muth)
        psi0_onemuth = digamma(onemuth)
        psi1_muth = polygamma(1, muth)
        psi1_onemuth = polygamma(1, onemuth)
        psi2_muth = polygamma(2, muth)
        psi2_onemuth = polygamma(2, onemuth)
        psi3_muth = polygamma(3, muth)
        psi3_onemuth = polygamma(3, onemuth)
        log_yoney = np.log(y) - np.log1p(-y)
        out = {
            "Dmu": 2.0 * wt * theta_value * (psi0_muth - psi0_onemuth - log_yoney),
            "Dmu2": 2.0 * wt * theta_value**2 * (psi1_muth + psi1_onemuth),
            "EDmu2": 2.0 * wt * theta_value**2 * (psi1_muth + psi1_onemuth),
        }
        if level > 0:
            out["Dth"] = (
                2.0
                * wt
                * theta_value
                * (
                    -mu * log_yoney
                    - np.log1p(-y)
                    + mu * psi0_muth
                    + onemu * psi0_onemuth
                    - psi0_th
                )
            )
            out["Dmuth"] = out["Dmu"] + 2.0 * wt * theta_value**2 * (
                mu * psi1_muth - onemu * psi1_onemuth
            )
            out["Dmu3"] = 2.0 * wt * theta_value**3 * (psi2_muth - psi2_onemuth)
            out["Dmu2th"] = 2.0 * out["Dmu2"] + 2.0 * wt * theta_value**3 * (
                mu * psi2_muth + onemu * psi2_onemuth
            )
            out["EDmu2th"] = out["Dmu2th"]
        if level > 1:
            out["Dmu4"] = 2.0 * wt * theta_value**4 * (psi3_muth + psi3_onemuth)
            out["Dth2"] = out["Dth"] + 2.0 * wt * theta_value**2 * (
                mu**2 * psi1_muth + onemu**2 * psi1_onemuth - psi1_th
            )
            out["Dmuth2"] = out["Dmuth"] + 2.0 * wt * theta_value**2 * (
                mu**2 * theta_value * psi2_muth
                + 2.0 * mu * psi1_muth
                - theta_value * onemu**2 * psi2_onemuth
                - 2.0 * onemu * psi1_onemuth
            )
            out["Dmu2th2"] = 2.0 * out["Dmu2th"] + 2.0 * wt * theta_value**3 * (
                mu**2 * theta_value * psi3_muth
                + 3.0 * mu * psi2_muth
                + onemu**2 * theta_value * psi3_onemuth
                + 3.0 * onemu * psi2_onemuth
            )
            out["Dmu3th"] = 3.0 * out["Dmu3"] + 2.0 * wt * theta_value**4 * (
                mu * psi3_muth - onemu * psi3_onemuth
            )
        return out
