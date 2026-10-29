"""Mixed-response extensions using the existing joint distributional engine."""

import numpy as np
from scipy.special import logit

from ....distributions.mixtures import zero_adjusted_gamma_prediction
from ._taylor import Taylor
from .distributional import _DistributionalFamily


class ZeroAdjustedGamma(_DistributionalFamily):
    """ZAGA(mu, sigma, pi0): positive-component mean/CV and zero atom mass.

    Natural/predictor columns are (mu,sigma,pi0), with log/log/logit links.
    gamlss.dist::ZAGA calls the atom probability nu. The whole response mean
    is (1-pi0)*mu. Finite-logit fitting requires both zero and positive rows;
    exact zero/one mixture weights are probability-only boundary configurations.
    Fixed smoothing and joint REML with Newton, BFGS and EFS are validated.
    Other method labels retain general-family REML coercion. Alternate links,
    censoring/truncation and deviance residuals remain unsupported.
    """

    name = "zaga_lss"
    parameter_names = ("mu", "sigma", "pi0")
    link_names = ("log", "log", "logit")
    parameter_bounds = {"mu": (0, None), "sigma": (0, None), "pi0": (0, 1)}
    max_derivative_order = 4
    supports_ml = supports_reml = supports_laml = True

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0):
            raise ValueError("zaga_lss requires finite nonnegative scalar responses.")
        return y

    def _log_density(self, y, log_mu, log_sigma, eta_zero):
        # gamlss.dist/R/ZAGA.R dZAGA. Evaluate each branch only on its rows,
        # retaining the normalized positive Gamma law and exact atom log mass.
        positive = y > 0
        atom = -(-eta_zero.subset(~positive)).softplus()
        lm, ls, lz = (v.subset(positive) for v in (log_mu, log_sigma, eta_zero))
        shape = (-2 * ls).exp()
        log_scale = lm + 2 * ls
        log_y = np.log(y[positive])
        density = (
            (shape - 1) * log_y
            - y[positive] * (-log_scale).exp()
            - shape * log_scale
            - shape.lgamma()
            - lz.softplus()
        )
        return Taylor.merge(positive, density, atom)

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y = self.validate_y(y)
        active = (
            np.ones(len(y), dtype=bool) if weights is None else np.asarray(weights) > 0
        )
        positive = y[active & (y > 0)]
        if not np.any(active & (y == 0)) or len(positive) < 2 or np.ptp(positive) == 0:
            raise ValueError(
                "zaga_lss fitting requires positive-weight zeros and at least two distinct positive responses; absent or degenerate components are unsupported."
            )
        return super().initialize(y, X, jj, offset=offset, weights=weights, E=E)

    def _initial_predictors(self, y):
        positive = y[y > 0]
        mean = np.mean(positive)
        return (
            np.log(np.where(y > 0, y, mean)),
            np.log(max(np.std(positive) / mean, 0.1)),
            logit(np.mean(y == 0)),
        )

    def logpdf_from_parameters(self, y, parameters):
        return zero_adjusted_gamma_prediction(
            self._check_parameters(parameters)
        ).log_prob(self.validate_y(y))

    def point_prediction_from_parameters(self, parameters):
        mu, sigma, pi0 = self._check_parameters(parameters).T
        return (1 - pi0) * mu

    def variance_from_parameters(self, parameters):
        mu, sigma, pi0 = self._check_parameters(parameters).T
        return (1 - pi0) * mu**2 * (sigma**2 + pi0)

    def rd(self, mu, wt, scale, *, rng):
        return zero_adjusted_gamma_prediction(mu).sample(n_draws=1, random_state=rng)[0]
