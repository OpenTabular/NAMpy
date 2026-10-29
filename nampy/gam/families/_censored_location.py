"""Shared response/link state for censored location-scale extended families."""

import numpy as np

from ..response import normalize_censored_response
from .family_base import ExtendedFamily


def unpack_censored(y):
    response = normalize_censored_response(y)
    return response.values, response.bounds


class CensoredLocationFamily(ExtendedFamily):
    """Common state only; subclasses supply source likelihood derivatives."""

    pirls_curvature_mode = "observed"

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        return np.exp(self._theta_working) if trans else self._theta_working.copy()

    def putTheta(self, theta):
        value = np.asarray(theta, float).reshape(-1)
        if value.size != 1 or not np.all(np.isfinite(value)):
            raise ValueError(f"{self.name} working theta must contain one finite value.")
        self._theta_working = value.copy()

    def validate_y(self, y):
        return normalize_censored_response(y)

    def initialize_mu(self, y, weights=None):
        del weights
        values = unpack_censored(y)[0]
        return (
            values.copy()
            if self.link_name == "identity"
            else np.maximum(values, np.min(values > 0))
        )

    def initialize_linear_predictors(self, y):
        return [self.link(self.initialize_mu(y))]

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def valid_mu(self, mu):
        mu = np.asarray(mu)
        return bool(
            np.all(np.isfinite(mu)) if self.link_name == "identity" else np.all(mu > 0)
        )

    def valid_eta(self, eta):
        eta = np.asarray(eta)
        return (
            bool(np.all(np.isfinite(eta)) and np.all(eta >= 0))
            if self.link_name == "sqrt"
            else True
        )

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        return 1.0

    def _state(self, y, weights=None, theta=None):
        values, bounds = unpack_censored(y)
        w = (
            np.ones_like(values)
            if weights is None
            else np.broadcast_to(np.asarray(weights, float), values.shape)
        )
        theta = (
            self.getTheta() if theta is None else np.asarray(theta, float).reshape(-1)
        )
        if theta.size != 1:
            raise ValueError(f"{self.name} working theta must have length one.")
        with np.errstate(divide="ignore", invalid="ignore"):
            th = theta[0] - np.log(w) / 2
        return values, bounds, th
