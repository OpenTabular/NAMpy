"""Four-predictor inflated beta with an interior-baseline probability simplex."""

import numpy as np
from scipy.special import expit, logit, softmax

from ....distributions.mixtures import (
    inflated_beta_prediction,
    validate_mixture_weights,
)
from ._taylor import Taylor
from .distributional import _DistributionalFamily


class InflatedBeta(_DistributionalFamily):
    """BEINF law with interior mean mu, precision phi, and two atom log ratios.

    Formula keys/predictors are (mu,phi,pi0,pi1). Raw columns mean
    (logit(mu),log(phi),log(pi0/pi_interior),log(pi1/pi_interior)). The last
    two are coupled baseline-category logits, not independent probability logits.
    Natural outputs are (mu,phi,pi0,pi1,pi_interior). The interior component is
    the reference category; it has no independently estimated predictor.
    It is derived from both atom predictors and has no separate additive term.

    R BEINF uses sigma=1/sqrt(1+phi), nu=pi0/pi_interior, tau=pi1/pi_interior.
    All three components and distinct interior responses are required for a
    finite-predictor fit. No absent-component or boundary fit is substituted.
    Alternate links, censoring/truncation and deviance residuals
    remain unsupported. Fixed smoothing and automatic joint REML with Newton,
    BFGS and EFS are validated. Other method labels retain general-family REML
    coercion; optim/magic/nlm.fd and separate ML/GCV/UBRE/NCV are unsupported.
    """

    name = "beinf_lss"
    parameter_names = ("mu", "phi", "pi0", "pi1")
    distribution_parameter_names = ("mu", "phi", "pi0", "pi1", "pi_interior")
    link_names = ("logit", "log", "identity", "identity")
    parameter_bounds = {
        "mu": (0, 1),
        "phi": (0, None),
        "pi0": (0, 1),
        "pi1": (0, 1),
        "pi_interior": (0, 1),
    }
    max_derivative_order = 4
    supports_ml = supports_reml = supports_laml = True

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("beinf_lss requires scalar responses inside [0,1].")
        return y

    def _check_parameters(self, parameters):
        parameters = super()._check_parameters(parameters)
        validate_mixture_weights(parameters[:, 2:4], parameters[:, 4])
        return parameters

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.ndim != 2 or eta.shape[1] != 4:
            raise ValueError("beinf_lss expects eta with shape (n, 4).")
        # Interior is the zero-logit reference. Stable for large finite logits;
        # preserve its weight explicitly, rather than subtracting atom weights.
        probabilities = softmax(
            np.column_stack([eta[:, 2:4], np.zeros(len(eta))]), axis=1
        )
        finite_rows = np.all(np.isfinite(eta), axis=1)
        if np.any(
            (probabilities[finite_rows] <= 0) | (probabilities[finite_rows] >= 1)
        ):
            raise NotImplementedError(
                "beinf_lss finite-logit probabilities exceed the float64 interior range; no boundary approximation is substituted. Boundary weights remain available through the shared probability law."
            )
        return np.column_stack([expit(eta[:, 0]), np.exp(eta[:, 1]), probabilities])

    def distribution_parameter_jacobian(self, eta):
        mu, phi, p0, p1, pc = self.distribution_parameters_from_eta(eta).T
        result = np.zeros((len(mu), 5, 4))
        result[:, 0, 0] = mu * (1 - mu)
        result[:, 1, 1] = phi
        result[:, 2, 2], result[:, 2, 3] = p0 * (1 - p0), -p0 * p1
        result[:, 3, 2], result[:, 3, 3] = -p1 * p0, p1 * (1 - p1)
        result[:, 4, 2], result[:, 4, 3] = -pc * p0, -pc * p1
        return result


    def _log_density(self, y, eta_mu, log_phi, eta_zero, eta_one):
        # gamlss.dist/R/BEINF.R dBEINF after phi=1/sigma**2-1. Shift the
        # reference log-sum-exp before Taylor evaluation: this identity is exact
        # for every local jet and keeps exponentials bounded by one.
        shift = np.maximum.reduce([np.zeros(len(y)), eta_zero.value, eta_one.value])
        log_normalizer = (
            (eta_zero - shift).exp() + (eta_one - shift).exp() + np.exp(-shift)
        ).log() + shift
        interior = (y > 0) & (y < 1)
        lm, lp = eta_mu.subset(interior), log_phi.subset(interior)
        mu, phi = 1 / (1 + (-lm).exp()), lp.exp()
        a, b = mu * phi, (1 - mu) * phi
        yy = y[interior]
        density = (
            phi.lgamma()
            - a.lgamma()
            - b.lgamma()
            + (a - 1) * np.log(yy)
            + (b - 1) * np.log1p(-yy)
            - log_normalizer.subset(interior)
        )
        endpoint = ~interior
        endpoint_logit = Taylor.merge(
            y[endpoint] == 0, eta_zero.subset(y == 0), eta_one.subset(y == 1)
        )
        return Taylor.merge(
            interior, density, endpoint_logit - log_normalizer.subset(endpoint)
        )

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y = self.validate_y(y)
        active = (
            np.ones(len(y), dtype=bool) if weights is None else np.asarray(weights) > 0
        )
        interior = y[active & (y > 0) & (y < 1)]
        if (
            not np.any(active & (y == 0))
            or not np.any(active & (y == 1))
            or len(interior) < 2
            or np.ptp(interior) == 0
        ):
            raise ValueError(
                "beinf_lss requires positive-weight zeros, ones and at least two distinct interior responses; absent or degenerate components are unsupported."
            )
        return super().initialize(y, X, jj, offset=offset, weights=weights, E=E)

    def _initial_predictors(self, y):
        interior = (y > 0) & (y < 1)
        yy = y[interior]
        mean = np.mean(yy)
        phi = max(mean * (1 - mean) / max(np.var(yy), 1e-6) - 1, 1)
        target = logit(np.where(interior, y, mean))
        count = np.sum(interior)
        return (
            target,
            np.log(phi),
            np.log(np.sum(y == 0) / count),
            np.log(np.sum(y == 1) / count),
        )

    def logpdf_from_parameters(self, y, parameters):
        return inflated_beta_prediction(self._check_parameters(parameters)).log_prob(
            self.validate_y(y)
        )

    def point_prediction_from_parameters(self, parameters):
        mu, phi, p0, p1, pc = self._check_parameters(parameters).T
        return p1 + pc * mu

    def variance_from_parameters(self, parameters):
        mu, phi, p0, p1, pc = self._check_parameters(parameters).T
        mean = p1 + pc * mu
        return (
            pc * (mu * (1 - mu) / (1 + phi) + (mu - mean) ** 2)
            + p0 * mean**2
            + p1 * (1 - mean) ** 2
        )

    def rd(self, mu, wt, scale, *, rng):
        return inflated_beta_prediction(mu).sample(n_draws=1, random_state=rng)[0]
