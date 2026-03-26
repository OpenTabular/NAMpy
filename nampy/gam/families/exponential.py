"""
Exponential-family GLM implementations.

Each class implements the :class:`~nampy.gam.families.base.GLMFamily` interface
for a specific distribution and canonical or common link function.  The classes
are used directly as ``family`` arguments to GAM model constructors.
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .base import GLMFamily, _EPS


class GaussianIdentityFamily(GLMFamily):
    name = "gaussian"
    link_name = "identity"
    canonical_link = True

    supports_closed_form_solve = True
    supports_pirls = True

    supports_gcv = True
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = True
    known_scale = None
    max_derivative_order = 1

    def initialize_mu(self, y):
        return np.asarray(y, dtype=np.float64).copy()

    def link(self, mu):
        return np.asarray(mu, dtype=np.float64)

    def inverse_link(self, eta):
        return np.asarray(eta, dtype=np.float64)

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.ones_like(eta)

    def variance(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.ones_like(mu)

    def dvar(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d2var(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3var(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def deviance(self, y, mu):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        return float(np.sum((y - mu) ** 2))

    def estimate_dispersion(self, y, mu, edf=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        rss = float(np.sum((y - mu) ** 2))
        if edf is None:
            return rss / max(len(y), 1.0)
        return rss / max(len(y) - float(edf), 1.0)

    def loglik_obs(self, y, mu, scale=1.0):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        scale = float(max(scale, self.eps))
        return (
            -0.5 * np.log(2.0 * np.pi * scale)
            - 0.5 * ((y - mu) ** 2) / scale
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        scale = float(max(scale, self.eps))
        mask = weights > 0
        nobs = int(np.sum(mask))
        return float(
            -0.5 * nobs * np.log(2.0 * np.pi * scale)
            + 0.5 * np.sum(np.log(weights[mask]))
        )



class BinomialLogitFamily(GLMFamily):
    name = "binomial"
    link_name = "logit"
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

    known_scale = 1.0
    max_derivative_order = 1

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialLogitFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip((y + 0.5) / 2.0, self.eps, 1.0 - self.eps)

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.log(mu / (1.0 - mu))

    def inverse_link(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def mu_eta(self, eta):
        mu = self.inverse_link(eta)
        return np.clip(mu * (1.0 - mu), self.eps, None)

    def variance(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.clip(mu * (1.0 - mu), self.eps, None)

    def dvar(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return 1.0 - 2.0 * mu

    def d2var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return -2.0 * np.ones_like(mu)

    def d3var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.zeros_like(mu)

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = np.clip((mu ** 2) * ((1.0 - mu) ** 2), self.eps, None)
        return (2.0 * mu - 1.0) / denom

    def d3link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = np.clip((mu**3) * ((1.0 - mu) ** 3), self.eps, None)
        return 2.0 * (3.0 * mu**2 - 3.0 * mu + 1.0) / denom

    def d4link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = np.clip((mu**4) * ((1.0 - mu) ** 4), self.eps, None)
        return 6.0 * (4.0 * mu**3 - 6.0 * mu**2 + 4.0 * mu - 1.0) / denom

    def deviance(self, y, mu):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        term1 = np.zeros_like(y, dtype=np.float64)
        mask1 = y > 0
        term1[mask1] = y[mask1] * np.log(y[mask1] / mu[mask1])
        term2 = np.zeros_like(y, dtype=np.float64)
        mask2 = y < 1
        term2[mask2] = (1.0 - y[mask2]) * np.log((1.0 - y[mask2]) / (1.0 - mu[mask2]))
        return float(2.0 * np.sum(term1 + term2))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)

        term = np.zeros_like(y, dtype=np.float64)
        mask1 = y > 0.0
        term[mask1] += y[mask1] * np.log(y[mask1])
        mask2 = y < 1.0
        term[mask2] += (1.0 - y[mask2]) * np.log(1.0 - y[mask2])
        return float(np.sum(weights * term))

    def working_weight_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = np.clip(mu * (1.0 - mu), self.eps, None)
        return (1.0 - 2.0 * mu) * W

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = np.clip(mu * (1.0 - mu), self.eps, None)
        return W * ((1.0 - 2.0 * mu) ** 2 - 2.0 * W)


class PoissonLogFamily(GLMFamily):
    name = "poisson"
    link_name = "log"
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

    known_scale = 1.0
    max_derivative_order = 1

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("PoissonLogFamily requires non-negative targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip(y + 0.1, self.eps, None)

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.log(mu)

    def inverse_link(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.exp(np.clip(eta, -30.0, 30.0))

    def mu_eta(self, eta):
        mu = self.inverse_link(eta)
        return np.clip(mu, self.eps, None)

    def variance(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.clip(mu, self.eps, None)

    def dvar(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.ones_like(mu)

    def d2var(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d3var(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return np.zeros_like(mu)

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

    def d3link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 / np.clip(mu**3, self.eps, None)

    def d4link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -6.0 / np.clip(mu**4, self.eps, None)

    def deviance(self, y, mu):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        term = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term[mask] = y[mask] * np.log(np.clip(y[mask], self.eps, None) / mu[mask])
        return float(2.0 * np.sum(term - (y - mu)))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return y * np.log(mu) - mu - gammaln(y + 1.0)

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        y_safe = np.clip(y, self.eps, None)
        sat = np.where(y > 0.0, y * np.log(y_safe) - y, -y) - gammaln(y + 1.0)
        return float(np.sum(weights * sat))

    def working_weight_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)


class GammaLogFamily(GLMFamily):
    name = "gamma"
    link_name = "log"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = True
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = False
    # Exact PIRLS derivatives for Gamma now rely on analytic working-weight
    # expressions that depend on the observations.
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True

    known_scale = None
    max_derivative_order = 1

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y <= 0.0):
            raise ValueError("GammaLogFamily requires strictly positive targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip(y, self.eps, None)

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.log(mu)

    def inverse_link(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.exp(np.clip(eta, -30.0, 30.0))

    def mu_eta(self, eta):
        mu = self.inverse_link(eta)
        return np.clip(mu, self.eps, None)

    def variance(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.clip(mu**2, self.eps, None)

    def dvar(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 * mu

    def d2var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 * np.ones_like(mu)

    def d3var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.zeros_like(mu)

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

    def d3link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 / np.clip(mu**3, self.eps, None)

    def d4link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -6.0 / np.clip(mu**4, self.eps, None)

    def deviance(self, y, mu):
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return float(2.0 * np.sum((y - mu) / mu - np.log(y / mu)))

    def estimate_dispersion(self, y, mu, edf=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        pearson = float(np.sum((y - mu) ** 2 / self.variance(mu)))
        if edf is None:
            return pearson / max(len(y), 1.0)
        return pearson / max(len(y) - float(edf), 1.0)

    def loglik_obs(self, y, mu, scale=1.0):
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        scale = float(max(scale, self.eps))
        shape = 1.0 / scale
        return (
            (shape - 1.0) * np.log(y)
            - y * shape / mu
            - gammaln(shape)
            - shape * np.log(mu / shape)
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del n
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        if weights is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        scale = float(max(scale, self.eps))
        shape = 1.0 / scale
        sat = -np.log(y) - shape - gammaln(shape) + shape * np.log(shape)
        return float(np.sum(weights * sat))

    def working_weight_derivative_eta(self, eta, y=None):
        if y is None:
            raise ValueError("GammaLogFamily requires targets to evaluate working-weight derivatives.")
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        return -np.asarray(y / mu, dtype=np.float64)

    def working_weight_second_derivative_eta(self, eta, y=None):
        if y is None:
            raise ValueError("GammaLogFamily requires targets to evaluate working-weight derivatives.")
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        return np.asarray(y / mu, dtype=np.float64)


class NegativeBinomialLogFamily(GLMFamily):
    """Fixed-theta NB2 family: Var(Y) = mu + mu^2 / theta"""

    name = "negbin"
    link_name = "log"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = False
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    def __init__(self, theta=1.0, estimate_theta=False, eps: float = _EPS):
        super().__init__(eps=eps)
        self.theta = float(theta)
        self.estimate_theta = bool(estimate_theta)
        if self.theta <= 0:
            raise ValueError("NegativeBinomialLogFamily requires theta > 0.")

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("NegativeBinomialLogFamily requires non-negative targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        # Match common negative-binomial initialization (MASS-style).
        return np.clip(y + (y == 0.0).astype(np.float64) / 6.0, self.eps, None)

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.log(mu)

    def inverse_link(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.exp(np.clip(eta, -30.0, 30.0))

    def mu_eta(self, eta):
        mu = self.inverse_link(eta)
        return np.clip(mu, self.eps, None)

    def variance(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.clip(mu + (mu**2) / self.theta, self.eps, None)

    def dvar(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 1.0 + 2.0 * mu / self.theta

    def d2var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return (2.0 / self.theta) * np.ones_like(mu)

    def d3var(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.zeros_like(mu)

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

    def d3link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 2.0 / np.clip(mu**3, self.eps, None)

    def d4link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -6.0 / np.clip(mu**4, self.eps, None)

    def deviance(self, y, mu):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        th = self.theta
        term1 = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term1[mask] = y[mask] * np.log(y[mask] / mu[mask])
        term2 = (y + th) * np.log((y + th) / (mu + th))
        return float(2.0 * np.sum(term1 - term2))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        th = float(self.theta)
        return (
            gammaln(y + th)
            - gammaln(th)
            - gammaln(y + 1.0)
            + th * np.log(th / (th + mu))
            + y * np.log(mu / (th + mu))
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
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

    def working_weight_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        th = float(self.theta)
        denom = np.clip(th + mu, self.eps, None)
        if y is None:
            # Fisher-weight derivative fallback: d/deta [th*mu/(th+mu)].
            return (th**2) * mu / np.clip(denom**2, self.eps, None)
        y = np.asarray(y, dtype=np.float64)
        # Exact derivative of P-IRLS Newton working weights w.r.t. eta:
        # w = wf * alpha, with wf = th*mu/(th+mu),
        # alpha = 1 + (y-mu)/(th+mu).
        num = mu * th * (th + y) * (th - mu)
        return num / np.clip(denom**3, self.eps, None)

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        th = float(self.theta)
        denom = np.clip(th + mu, self.eps, None)
        if y is None:
            # Fisher-weight second derivative fallback.
            return (th**2) * mu * (th - mu) / np.clip(denom**3, self.eps, None)
        y = np.asarray(y, dtype=np.float64)
        # Exact second derivative of P-IRLS Newton working weights w.r.t. eta.
        num = mu * th * (th + y) * (mu**2 - 4.0 * mu * th + th**2)
        return num / np.clip(denom**4, self.eps, None)

    def estimate_theta_mle(self, y, mu, weights=None, max_iter=50, tol=1e-7):
        """
        MLE of the NB dispersion parameter theta given current mu.

        Optimises the NB log-likelihood over theta using Newton-Raphson on
        the log(theta) scale to keep theta > 0.  The gradient and Hessian
        w.r.t. log(theta) = phi are:

          g(phi) = theta * d ell / d theta
          H(phi) = theta^2 * d^2 ell / d theta^2 + theta * d ell / d theta
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        w = (
            np.ones_like(y, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )

        theta = max(float(self.theta), self.eps)
        for _ in range(max_iter):
            # first derivative of ell w.r.t. theta
            score = float(np.sum(
                w * (
                    digamma(y + theta)
                    - digamma(theta)
                    + np.log(theta / (theta + mu))
                    + 1.0
                    - (y + theta) / (mu + theta)
                )
            ))
            # second derivative of ell w.r.t. theta
            hess = float(np.sum(
                w * (
                    polygamma(1, y + theta)
                    - polygamma(1, theta)
                    + mu / (theta * (theta + mu))
                    + (y - mu) / (theta + mu) ** 2
                )
            ))

            if not np.isfinite(score) or not np.isfinite(hess):
                break

            # Newton on log(theta) = phi: phi_new = phi - g/H where
            # g = theta * score, H = theta^2 * hess + theta * score
            g_phi = theta * score
            h_phi = theta ** 2 * hess + theta * score
            if h_phi == 0.0 or not np.isfinite(h_phi):
                break

            phi = np.log(max(theta, self.eps))
            phi_new = phi - g_phi / h_phi
            theta_new = np.exp(np.clip(phi_new, -20.0, 10.0))

            if abs(theta_new - theta) < tol * (1.0 + abs(theta)):
                theta = theta_new
                break
            theta = theta_new

        return max(theta, self.eps)
