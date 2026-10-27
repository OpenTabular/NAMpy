import numpy as np
from scipy.stats import poisson as poisson_distribution

from ._r_poisson_math import log_dpois
from .family_base import GLMFamily


class PoissonLogFamily(GLMFamily):
    """Poisson family with log link. Matches mgcv::poisson(link="log")."""

    name = "poisson"
    link_name = "log"
    canonical_link = True

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "log"
    _variance_key = "poisson"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("PoissonLogFamily requires non-negative targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return y + 0.1

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0.0))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return bool(np.all(np.isfinite(eta)))

    def deviance(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        weights = self._check_weights(y, weights)
        term = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term[mask] = y[mask] * np.log(y[mask] / mu[mask])
        return float(2.0 * np.sum(weights * (term - (y - mu))))

    def deviance_obs(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        weights = self._check_weights(y, weights)
        term = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        term[mask] = y[mask] * np.log(y[mask] / mu[mask])
        return 2.0 * weights * (term - (y - mu))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        return log_dpois(y, mu)

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        weights = self._check_weights(y, weights)
        return float(np.sum(weights * log_dpois(y, y)))

    def working_weight_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)

    def quantile_residual_bounds(self, y, mu, *, weights=None, scale=1.0):
        del weights, scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        lower = poisson_distribution.cdf(y - 1.0, mu=mu)
        upper = poisson_distribution.cdf(y, mu=mu)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


class PoissonIdentityFamily(PoissonLogFamily):
    """Poisson family with identity link. Matches mgcv::poisson(link="identity")."""

    link_name = "identity"
    canonical_link = False
    _link_key = "identity"

    def validate_y(self, y):
        y = GLMFamily.validate_y(self, y)
        if np.any(y < 0.0):
            raise ValueError("PoissonIdentityFamily requires non-negative targets.")
        return y

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return bool(np.all(np.isfinite(eta)))

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        if y is not None:
            return -2.0 * np.asarray(y, dtype=float) / eta**3
        return -1.0 / eta**2

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        if y is not None:
            return 6.0 * np.asarray(y, dtype=float) / eta**4
        return 2.0 / eta**3


class PoissonSqrtFamily(PoissonLogFamily):
    """Poisson family with sqrt link. Matches mgcv::poisson(link="sqrt")."""

    link_name = "sqrt"
    canonical_link = False
    _link_key = "sqrt"

    def validate_y(self, y):
        y = GLMFamily.validate_y(self, y)
        if np.any(y < 0.0):
            raise ValueError("PoissonSqrtFamily requires non-negative targets.")
        return y

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0.0))

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        if y is not None:
            return -4.0 * np.asarray(y, dtype=float) / eta**3
        return np.zeros_like(eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        if y is not None:
            return 12.0 * np.asarray(y, dtype=float) / eta**4
        return np.zeros_like(eta)
