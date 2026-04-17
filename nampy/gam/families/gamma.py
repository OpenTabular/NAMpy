import numpy as np

from .family_base import _GammaBase


class GammaLogFamily(_GammaBase):
    """Gamma family with log link. Matches mgcv::Gamma(link="log")."""

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

    _link_key = "log"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y <= 0.0):
            raise ValueError("GammaLogFamily requires strictly positive targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip(y, self.eps, None)

    def working_weight_derivative_eta(self, eta, y=None):
        if y is None:
            raise ValueError(
                "GammaLogFamily requires targets to evaluate working-weight derivatives."
            )
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        return -np.asarray(y / mu, dtype=np.float64)

    def working_weight_second_derivative_eta(self, eta, y=None):
        if y is None:
            raise ValueError(
                "GammaLogFamily requires targets to evaluate working-weight derivatives."
            )
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        return np.asarray(y / mu, dtype=np.float64)


class GammaIdentityFamily(_GammaBase):
    """Gamma family with identity link. Matches mgcv::Gamma(link="identity")."""

    name = "gamma"
    link_name = "identity"
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

    _link_key = "identity"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y <= 0.0):
            raise ValueError("GammaIdentityFamily requires strictly positive targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip(y, self.eps, None)

    def working_weight_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        return -2.0 / np.clip(mu**3, self.eps, None)

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        return 6.0 / np.clip(mu**4, self.eps, None)


class GammaInverseFamily(_GammaBase):
    """Gamma family with inverse link. Matches mgcv::Gamma(link="inverse")."""

    name = "gamma"
    link_name = "inverse"
    canonical_link = True

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

    _link_key = "inverse"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y <= 0.0):
            raise ValueError("GammaInverseFamily requires strictly positive targets.")
        return y

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip(y, self.eps, None)

    def working_weight_derivative_eta(self, eta, y=None):
        # W = mu^2 (canonical link, W_exact = W_Fisher = mu^2).
        # dW/deta = 2*mu * dmu/deta = 2/eta * (-1/eta^2) = -2/eta^3
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        return -2.0 * mu**3

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = np.clip(self.inverse_link(eta), self.eps, None)
        return 6.0 * mu**4
