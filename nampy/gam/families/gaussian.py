import numpy as np

from .family_base import GLMFamily


class GaussianIdentityFamily(GLMFamily):
    """Gaussian family with identity link. Matches mgcv::gaussian()."""

    name = "gaussian"
    link_name = "identity"
    canonical_link = True

    supports_closed_form_solve = True
    supports_pirls = True

    supports_gcv = True
    supports_ncv = True
    supports_qncv = True
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = False
    known_scale = None
    max_derivative_order = 1

    _link_key = "identity"
    _variance_key = "constant"

    def initialize_mu(self, y):
        return np.asarray(y, dtype=np.float64).copy()

    def deviance(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        weights = self._check_weights(y, weights)
        return float(np.sum(weights * (y - mu) ** 2))

    def deviance_obs(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        w = self._check_weights(y, weights)
        return w * (y - mu) ** 2

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        if self.known_scale is not None:
            return float(self.known_scale)
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        w = self._check_weights(y, weights)
        rss = float(np.sum(w * (y - mu) ** 2))
        # Use n (number of observations) in denominator to match mgcv/glm convention.
        # mgcv divides by (n - edf), not (sum(w) - edf), for Gaussian scale estimation.
        n = float(y.shape[0])
        if edf is None:
            return rss / n
        return rss / (n - float(edf))

    def loglik_obs(self, y, mu, scale=1.0):
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        scale = float(max(scale, self.eps))
        return -0.5 * np.log(2.0 * np.pi * scale) - 0.5 * ((y - mu) ** 2) / scale

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        y = np.asarray(y, dtype=np.float64)
        weights = self._check_weights(y, weights)
        scale = float(max(scale, self.eps))
        mask = weights > 0
        nobs = int(np.sum(mask))
        return float(
            -0.5 * nobs * np.log(2.0 * np.pi * scale)
            + 0.5 * np.sum(np.log(weights[mask]))
        )

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return np.zeros_like(eta, dtype=np.float64)

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return np.zeros_like(eta, dtype=np.float64)


class GaussianLogFamily(GaussianIdentityFamily):
    """Gaussian family with log link. Matches mgcv::gaussian(link="log")."""

    link_name = "log"
    canonical_link = False

    supports_closed_form_solve = False
    max_derivative_order = 1

    _link_key = "log"

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.maximum(y, 0.01 * float(np.std(y, ddof=1)))

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0.0))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return bool(np.all(np.isfinite(eta)))

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return 2.0 * np.exp(2.0 * eta)

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return 4.0 * np.exp(2.0 * eta)


class GaussianInverseFamily(GaussianIdentityFamily):
    """Gaussian family with inverse link. Matches mgcv::gaussian(link="inverse")."""

    link_name = "inverse"
    canonical_link = False

    supports_closed_form_solve = False
    max_derivative_order = 1

    _link_key = "inverse"

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return y + (y == 0.0).astype(np.float64) * float(np.std(y, ddof=1)) * 0.01

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all(mu != 0.0))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return bool(np.all(np.isfinite(eta)) and np.all(eta != 0.0))

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return -4.0 / eta**5

    def working_weight_second_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        return 20.0 / eta**6
