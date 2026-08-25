import numpy as np
from scipy.stats import norm as _norm

from .family_base import _BinomialBase


class BinomialLogitFamily(_BinomialBase):
    """Binomial family with logit link. Matches mgcv::binomial(link="logit")."""

    name = "binomial"
    link_name = "logit"
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

    _link_key = "logit"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialLogitFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return np.clip((w * y + 0.5) / (w + 1.0), self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = mu * (1.0 - mu)
        return (1.0 - 2.0 * mu) * W

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = mu * (1.0 - mu)
        return W * ((1.0 - 2.0 * mu) ** 2 - 2.0 * W)


class BinomialProbitFamily(_BinomialBase):
    """Binomial family with probit link. Matches mgcv::binomial(link="probit")."""

    name = "binomial"
    link_name = "probit"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    # mgcv's REML/Newton outer loop uses exact probit d2link/d3link/d4link
    # derivatives from fix.family.link.family() in gam.fit3.r.
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "probit"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialProbitFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return np.clip((w * y + 0.5) / (w + 1.0), self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        phi = _norm.pdf(eta)
        mu = _norm.cdf(eta)
        V = mu * (1.0 - mu)
        # d/deta [phi^2/V] = phi^2 * [-2*eta*V - phi*(1-2*mu)] / V^2
        return phi**2 * (-2.0 * eta * V - phi * (1.0 - 2.0 * mu)) / (V**2)


class BinomialCloglogFamily(_BinomialBase):
    """Binomial family with cloglog link. Matches mgcv::binomial(link="cloglog")."""

    name = "binomial"
    link_name = "cloglog"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    # mgcv::fix.family.link.family() defines d2link/d3link/d4link for cloglog.
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "cloglog"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialCloglogFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return np.clip((w * y + 0.5) / (w + 1.0), self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(eta)
        mu = 1.0 - np.exp(-lam)
        M = lam * np.exp(-lam)
        V = mu * (1.0 - mu)
        # d/deta [M^2/V] = M^2 * [2*(1-lam)*V - (1-2*mu)*M] / V^2
        return M**2 * (2.0 * (1.0 - lam) * V - (1.0 - 2.0 * mu) * M) / (V**2)


class BinomialCauchitFamily(_BinomialBase):
    """Binomial family with cauchit link. Matches mgcv::binomial(link="cauchit")."""

    name = "binomial"
    link_name = "cauchit"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    # mgcv::fix.family.link.family() defines d2link/d3link/d4link for cauchit.
    supports_exact_pirls_second_derivatives = True

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "cauchit"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialCauchitFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return np.clip((w * y + 0.5) / (w + 1.0), self.eps, 1.0 - self.eps)


class BinomialLogFamily(_BinomialBase):
    """Binomial family with log link. Matches mgcv::binomial(link="log")."""

    name = "binomial"
    link_name = "log"
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

    known_scale = 1.0
    max_derivative_order = 1

    _link_key = "log"

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("BinomialLogFamily requires targets in [0, 1].")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return np.clip((w * y + 0.5) / (w + 1.0), self.eps, 1.0 - self.eps)
