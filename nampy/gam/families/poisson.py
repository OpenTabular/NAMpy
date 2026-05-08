import numpy as np
from scipy.special import gammaln

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
    supports_laml = False
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
        return y * np.log(mu) - mu - gammaln(y + 1.0)

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        weights = self._check_weights(y, weights)
        y_safe = np.clip(y, self.eps, None)
        sat = np.where(y > 0.0, y * np.log(y_safe) - y, -y) - gammaln(y + 1.0)
        return float(np.sum(weights * sat))

    def working_weight_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        return self.inverse_link(eta)


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
        return -1.0 / eta**2

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
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
        return np.zeros_like(eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return np.zeros_like(eta)
