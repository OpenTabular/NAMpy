import numpy as np
from scipy.stats import norm as _norm

from .._mgcv_constants import LINK_ETA_EXP_CLIP
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
    supports_laml = False
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

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip((y + 0.5) / 2.0, self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = np.clip(mu * (1.0 - mu), self.eps, None)
        return (1.0 - 2.0 * mu) * W

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        W = np.clip(mu * (1.0 - mu), self.eps, None)
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
    supports_laml = False
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

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip((y + 0.5) / 2.0, self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.clip(_norm.pdf(eta), self.eps, None)
        mu = np.clip(_norm.cdf(eta), self.eps, 1.0 - self.eps)
        V = np.clip(mu * (1.0 - mu), self.eps, None)
        # d/deta [phi^2/V] = phi^2 * [-2*eta*V - phi*(1-2*mu)] / V^2
        return (
            phi**2
            * (-2.0 * eta * V - phi * (1.0 - 2.0 * mu))
            / np.clip(V**2, self.eps, None)
        )


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
    supports_laml = False
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

    def initialize_mu(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.clip((y + 0.5) / 2.0, self.eps, 1.0 - self.eps)

    def working_weight_derivative_eta(self, eta, y=None):
        eta = np.asarray(eta, dtype=np.float64)
        lam = np.exp(np.clip(eta, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP))
        mu = np.clip(1.0 - np.exp(-lam), self.eps, 1.0 - self.eps)
        M = np.clip(lam * np.exp(-lam), self.eps, None)
        V = np.clip(mu * (1.0 - mu), self.eps, None)
        # d/deta [M^2/V] = M^2 * [2*(1-lam)*V - (1-2*mu)*M] / V^2
        return (
            M**2
            * (2.0 * (1.0 - lam) * V - (1.0 - 2.0 * mu) * M)
            / np.clip(V**2, self.eps, None)
        )
