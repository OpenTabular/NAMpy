import numpy as np
from scipy.special import gammaln

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

    def d2link(self, mu):
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

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        denom = np.clip((mu ** 2) * ((1.0 - mu) ** 2), self.eps, None)
        return (2.0 * mu - 1.0) / denom

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

    def working_weight_derivative_eta(self, eta):
        mu = self.inverse_link(eta)
        W = np.clip(mu * (1.0 - mu), self.eps, None)
        return (1.0 - 2.0 * mu) * W

    def working_weight_second_derivative_eta(self, eta):
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

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

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

    def working_weight_derivative_eta(self, eta):
        return self.inverse_link(eta)

    def working_weight_second_derivative_eta(self, eta):
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

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

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

    def working_weight_derivative_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.zeros_like(eta)

    def working_weight_second_derivative_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.zeros_like(eta)


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

    def __init__(self, theta=1.0, eps: float = _EPS):
        super().__init__(eps=eps)
        self.theta = float(theta)
        if self.theta <= 0:
            raise ValueError("NegativeBinomialLogFamily requires theta > 0.")

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("NegativeBinomialLogFamily requires non-negative targets.")
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
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return np.clip(mu + (mu**2) / self.theta, self.eps, None)

    def dvar(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return 1.0 + 2.0 * mu / self.theta

    def d2link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        return -1.0 / np.clip(mu ** 2, self.eps, None)

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
        y_safe = np.clip(y, self.eps, None)
        term = (
            gammaln(y + th)
            - gammaln(th)
            - gammaln(y + 1.0)
            + th * np.log(th / (th + y_safe))
        )
        mask = y > 0.0
        term[mask] += y[mask] * np.log(y_safe[mask] / (th + y_safe[mask]))
        return float(np.sum(weights * term))

    def working_weight_derivative_eta(self, eta):
        mu = self.inverse_link(eta)
        th = float(self.theta)
        denom = th + mu
        return (th ** 2) * mu / np.clip(denom ** 2, self.eps, None)

    def working_weight_second_derivative_eta(self, eta):
        mu = self.inverse_link(eta)
        th = float(self.theta)
        denom = th + mu
        return (th ** 2) * mu * (th - mu) / np.clip(denom ** 3, self.eps, None)
