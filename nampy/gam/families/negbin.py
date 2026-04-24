import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .._mgcv_constants import FAMILY_EPS
from ._function_maps import LINK_REGISTRY, NegativeBinomialVariance
from .family_base import ExtendedFamily


class NegativeBinomialLogFamily(ExtendedFamily):
    """Negative binomial family with log link. Matches mgcv::negbin(link="log")."""

    name = "negbin"
    link_name = "log"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ncv = True
    supports_qncv = True
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = False
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "log"

    def __init__(
        self,
        theta=1.0,
        estimate_theta=False,
        link: str = "log",
        eps: float = FAMILY_EPS,
    ):
        super().__init__(eps=eps)
        link_key = str(link).lower()
        if link_key not in {"log", "identity", "sqrt"}:
            raise ValueError(
                "Negative-binomial link must be one of 'identity', 'log', or 'sqrt'."
            )
        self._link_key = link_key
        self.link_name = link_key
        self.link = LINK_REGISTRY[self._link_key](eps=self.eps)
        self.theta = float(theta)
        self.estimate_theta = bool(estimate_theta)
        self.variance = NegativeBinomialVariance(eps=self.eps, family=self)

    @property
    def n_theta(self):
        return 1 if bool(self.estimate_theta) else 0

    @property
    def ini_theta(self):
        return float(self._log_theta)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        theta = float(value)
        if theta <= 0:
            raise ValueError("NegativeBinomialLogFamily requires theta > 0.")
        self._theta = theta
        self._log_theta = float(np.log(theta))

    def getTheta(self, trans=False):
        return float(self.theta) if trans else float(self._log_theta)

    def putTheta(self, theta):
        log_theta = float(theta)
        theta_val = float(np.exp(log_theta))
        if not np.isfinite(theta_val) or theta_val <= 0.0:
            raise ValueError("NegativeBinomialLogFamily requires finite log(theta).")
        self._theta = theta_val
        self._log_theta = log_theta

    def _check_weights(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            return np.ones_like(y, dtype=np.float64)
        return np.asarray(weights, dtype=np.float64)

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

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        del y, mu, edf, weights
        return float(self.known_scale)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("NegativeBinomialLogFamily requires non-negative targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        # Match common negative-binomial initialization (MASS-style).
        return np.clip(y + (y == 0.0).astype(np.float64) / 6.0, self.eps, None)

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0.0))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        if self.link_name in {"identity", "sqrt"}:
            return bool(np.all(np.isfinite(eta)) and np.all(eta > 0.0))
        return bool(np.all(np.isfinite(eta)))

    def deviance(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        weights = self._check_weights(y, weights)
        th = self.theta
        term1 = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term1[mask] = y[mask] * np.log(np.maximum(1.0, y[mask]) / mu[mask])
        term2 = (y + th) * np.log((y + th) / (mu + th))
        return float(2.0 * np.sum(weights * (term1 - term2)))

    def deviance_obs(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        weights = self._check_weights(y, weights)
        th = self.theta
        term1 = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term1[mask] = y[mask] * np.log(np.maximum(1.0, y[mask]) / mu[mask])
        term2 = (y + th) * np.log((y + th) / (mu + th))
        return 2.0 * weights * (term1 - term2)

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        th = float(self.theta)
        return (
            gammaln(y + th)
            - gammaln(th)
            - gammaln(y + 1.0)
            + th * np.log(th / (th + mu))
            + y * np.log(mu / (th + mu))
        )

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        weights = self._check_weights(y, weights)
        th = float(self.theta)
        # Saturated NB log-likelihood at y (count density at mean y):
        # keep y==0 exact (contributes 0) instead of epsilon-shifting.
        term = (
            gammaln(y + th)
            - gammaln(th)
            - gammaln(y + 1.0)
            + th * np.log(th / (th + y))
        )
        mask = y > 0.0
        term[mask] += y[mask] * np.log(y[mask] / (th + y[mask]))
        return float(np.sum(weights * term))

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        """
        Port of `mgcv::nb()$Dd`.

        `theta` is log(theta), matching upstream extended-family storage.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        wt = self._check_weights(y, wt)
        ltheta = float(self.getTheta(False) if theta is None else theta)
        theta_val = float(np.exp(ltheta))
        yth = y + theta_val
        muth = mu + theta_val

        out = {
            "Dmu": 2.0 * wt * (yth / muth - y / mu),
            "Dmu2": -2.0 * wt * (yth / (muth**2) - y / (mu**2)),
            "EDmu2": 2.0 * wt * (1.0 / mu - 1.0 / muth),
        }
        if level > 0:
            out["Dth"] = (
                -2.0 * wt * theta_val * (np.log(yth / muth) + (1.0 - yth / muth))
            )
            out["Dmuth"] = 2.0 * wt * theta_val * (1.0 - yth / muth) / muth
            out["Dmu3"] = 4.0 * wt * (yth / (muth**3) - y / (mu**3))
            out["Dmu2th"] = 2.0 * wt * theta_val * (2.0 * yth / muth - 1.0) / (muth**2)
            out["EDmu2th"] = 2.0 * wt / (muth**2)
        if level > 1:
            out["Dmu4"] = 2.0 * wt * (6.0 * y / (mu**4) - 6.0 * yth / (muth**4))
            out["Dth2"] = (
                -2.0
                * wt
                * theta_val
                * (
                    np.log(yth / muth)
                    + theta_val * yth / (muth**2)
                    - yth / muth
                    - 2.0 * theta_val / muth
                    + 1.0
                    + theta_val / yth
                )
            )
            out["Dmuth2"] = (
                2.0
                * wt
                * theta_val
                * (
                    2.0 * theta_val * yth / (muth**2)
                    - yth / muth
                    - 2.0 * theta_val / muth
                    + 1.0
                )
                / muth
            )
            out["Dmu2th2"] = (
                2.0
                * wt
                * theta_val
                * (
                    -6.0 * yth * theta_val / (muth**2)
                    + 2.0 * yth / muth
                    + 4.0 * theta_val / muth
                    - 1.0
                )
                / (muth**2)
            )
            out["Dmu3th"] = 4.0 * wt * theta_val * (1.0 - 3.0 * yth / muth) / (muth**3)
        return out

    def ls(self, y, w, theta=None, scale=1.0):
        """
        Port of `mgcv::nb()$ls`.

        Returns saturated log-likelihood and derivatives w.r.t. log(theta).
        """
        del scale
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, w)
        ltheta = float(self.getTheta(False) if theta is None else theta)
        theta_val = float(np.exp(ltheta))
        ylogy = np.zeros_like(y, dtype=np.float64)
        mask = y > 0.0
        ylogy[mask] = y[mask] * np.log(y[mask])
        term = (
            (y + theta_val) * np.log(np.clip(y + theta_val, self.eps, None))
            - ylogy
            + gammaln(y + 1.0)
            - theta_val * np.log(theta_val)
            + gammaln(theta_val)
            - gammaln(theta_val + y)
        )
        ls = -float(np.sum(term * w))

        yth = y + theta_val
        lyth = np.log(np.clip(yth, self.eps, None))
        psi0_yth = digamma(yth)
        psi0_th = digamma(theta_val)
        term1 = theta_val * (lyth - psi0_yth + psi0_th - ltheta)
        LSTH1 = np.asarray((-term1 * w)[:, None], dtype=np.float64)
        lsth1 = float(np.sum(LSTH1))

        psi1_yth = polygamma(1, yth)
        psi1_th = polygamma(1, theta_val)
        term2 = theta_val * (
            lyth
            - theta_val * psi1_yth
            - psi0_yth
            + theta_val / yth
            + theta_val * psi1_th
            + psi0_th
            - ltheta
            - 1.0
        )
        lsth2 = -float(np.sum(term2 * w))
        return {"ls": ls, "lsth1": lsth1, "LSTH1": LSTH1, "lsth2": lsth2}

    def working_weight_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        mu_eta = np.asarray(
            self.inverse_link_derivatives(eta, order=1), dtype=np.float64
        )
        mu_eta2 = np.asarray(
            self.inverse_link_derivatives(eta, order=2), dtype=np.float64
        )
        var = np.clip(np.asarray(self.variance(mu), dtype=np.float64), self.eps, None)
        dvar = np.asarray(self.dvar(mu), dtype=np.float64)
        if y is None:
            return 2.0 * mu_eta * mu_eta2 / var - (mu_eta**3) * dvar / np.clip(
                var**2, self.eps, None
            )

        y = np.asarray(y, dtype=np.float64)
        th = float(self.theta)
        alpha = 1.0 + (y - mu) / np.clip(th + mu, self.eps, None)
        fisher_deriv = 2.0 * mu_eta * mu_eta2 / var - (mu_eta**3) * dvar / np.clip(
            var**2, self.eps, None
        )
        alpha_deriv = -mu_eta * (th + y) / np.clip((th + mu) ** 2, self.eps, None)
        fisher_weight = (mu_eta**2) / var
        return fisher_deriv * alpha + fisher_weight * alpha_deriv

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        mu_eta = np.asarray(
            self.inverse_link_derivatives(eta, order=1), dtype=np.float64
        )
        mu_eta2 = np.asarray(
            self.inverse_link_derivatives(eta, order=2), dtype=np.float64
        )
        mu_eta3 = np.asarray(
            self.inverse_link_derivatives(eta, order=3), dtype=np.float64
        )
        var = np.clip(np.asarray(self.variance(mu), dtype=np.float64), self.eps, None)
        dvar = np.asarray(self.dvar(mu), dtype=np.float64)
        d2var = np.asarray(self.d2var(mu), dtype=np.float64)
        fisher_second = (
            2.0 * (mu_eta2**2 + mu_eta * mu_eta3) / var
            - (5.0 * (mu_eta**2) * mu_eta2 * dvar + (mu_eta**4) * d2var)
            / np.clip(var**2, self.eps, None)
            + 2.0 * (mu_eta**4) * (dvar**2) / np.clip(var**3, self.eps, None)
        )
        if y is None:
            return fisher_second

        y = np.asarray(y, dtype=np.float64)
        th = float(self.theta)
        denom = np.clip(th + mu, self.eps, None)
        alpha = 1.0 + (y - mu) / denom
        alpha_deriv = -mu_eta * (th + y) / np.clip(denom**2, self.eps, None)
        alpha_second = -(th + y) * mu_eta2 / np.clip(
            denom**2, self.eps, None
        ) + 2.0 * (th + y) * (mu_eta**2) / np.clip(denom**3, self.eps, None)
        fisher_weight = (mu_eta**2) / var
        fisher_deriv = 2.0 * mu_eta * mu_eta2 / var - (mu_eta**3) * dvar / np.clip(
            var**2, self.eps, None
        )
        return (
            fisher_second * alpha
            + 2.0 * fisher_deriv * alpha_deriv
            + fisher_weight * alpha_second
        )

    def estimate_theta_mle(self, y, mu, weights=None, max_iter=50, tol=1e-7):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        w = (
            np.ones_like(y, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )

        def _nll_grad_hess(log_theta_val: float):
            log_theta_val = float(log_theta_val)
            theta_val = max(float(np.exp(log_theta_val)), self.eps)
            yth = y + theta_val
            muth = mu + theta_val

            # mgcv::estimate.theta -> nlogl(): dev/2 - ls, all in log-theta.
            dev = float(
                np.sum(
                    2.0
                    * w
                    * (y * np.log(np.maximum(1.0, y) / mu) - yth * np.log(yth / muth))
                )
            )

            ylogy = y.copy()
            ind = y > 0.0
            ylogy[ind] = y[ind] * np.log(y[ind])
            ls_term = (
                yth * np.log(yth)
                - ylogy
                + gammaln(y + 1.0)
                - theta_val * log_theta_val
                + gammaln(theta_val)
                - gammaln(yth)
            )
            ls = -float(np.sum(ls_term * w))
            nll = dev / 2.0 - ls

            dth = -2.0 * w * theta_val * (np.log(yth / muth) + (1.0 - yth / muth))
            g1 = float(np.sum(dth) / 2.0)

            psi0_yth = digamma(yth)
            psi0_th = digamma(theta_val)
            lsth_term = theta_val * (np.log(yth) - psi0_yth + psi0_th - log_theta_val)
            lsth1 = -float(np.sum(lsth_term * w))
            grad = g1 - lsth1

            dth2 = (
                -2.0
                * w
                * theta_val
                * (
                    np.log(yth / muth)
                    + theta_val * yth / (muth**2)
                    - yth / muth
                    - 2.0 * theta_val / muth
                    + 1.0
                    + theta_val / yth
                )
            )
            Dth2 = float(np.sum(dth2) / 2.0)

            psi1_yth = polygamma(1, yth)
            psi1_th = polygamma(1, theta_val)
            lsth2_term = theta_val * (
                np.log(yth)
                - theta_val * psi1_yth
                - psi0_yth
                + theta_val / yth
                + theta_val * psi1_th
                + psi0_th
                - log_theta_val
                - 1.0
            )
            lsth2 = -float(np.sum(lsth2_term * w))
            hess = Dth2 - lsth2
            return nll, grad, hess

        theta = float(np.log(max(float(self.theta), self.eps)))
        nll, grad, hess = _nll_grad_hess(theta)
        if not np.isfinite(nll) or not np.isfinite(grad) or not np.isfinite(hess):
            return max(float(np.exp(theta)), self.eps)

        for _ in range(min(int(max_iter), 100)):
            if abs(grad) <= tol * (abs(nll) + 1.0):
                break
            hess_eff = float(hess)
            if not np.isfinite(hess_eff):
                break
            if hess_eff <= 0.0:
                hess_eff = max(abs(hess_eff), 1e-5)

            step = -grad / hess_eff
            if abs(step) > 4.0:
                step *= 4.0 / abs(step)

            theta_new = theta + step
            iter_halve = 0
            nll_new = np.inf
            grad_new = grad
            hess_new = hess
            while iter_halve <= 25:
                if not np.isfinite(theta_new):
                    nll_new = np.inf
                else:
                    nll_new, grad_new, hess_new = _nll_grad_hess(theta_new)
                if np.isfinite(nll_new) and nll_new - nll <= np.finfo(
                    np.float64
                ).eps ** 0.75 * abs(nll):
                    break
                step *= 0.5
                iter_halve += 1
                if abs(step) <= self.eps * max(1.0, abs(theta)):
                    theta_new = theta
                    nll_new, grad_new, hess_new = nll, grad, hess
                    break
                theta_new = theta + step

            if theta_new == theta:
                break
            theta = float(theta_new)
            nll, grad, hess = nll_new, grad_new, hess_new

        return max(float(np.exp(theta)), self.eps)
