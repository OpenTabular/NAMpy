"""Beta-binomial extension; normalized BB likelihood, not R working curvature."""

import numpy as np
from scipy.special import gammaln, logit

from ....distributions import BetaBinomialLaw, DistributionPrediction
from ....distributions.laws import validate_trials
from .distributional import _DistributionalFamily


class BetaBinomialLocationPrecision(_DistributionalFamily):
    """Count LSS with logit(mu), log(kappa), explicit context['trials'].

    Corresponds to gamlss.dist::BB(mu, sigma=1/kappa, bd=trials).
    Fixed smoothing and automatic joint REML with outer_newton, bfgs and efs
    are validated. Other method labels retain general-family REML coercion.
    Trials are nonnegative integers and are never prior weights.
    Positive-weight data must contain a row with at least two trials, and both
    successes and failures; these necessary guards do not guarantee full model
    identifiability. Alternate links, NCV and deviance residuals are unsupported.
    """

    name = "betabinomial_lss"
    parameter_names = ("mu", "kappa")
    link_names = ("logit", "log")
    parameter_bounds = {"mu": (0, 1), "kappa": (0, None)}
    context_names = ("trials",)
    max_derivative_order = 4
    supports_ml = supports_reml = supports_laml = True

    def bind_fit_context(self, context, n_rows, *, weights=None):
        super().bind_fit_context(context, n_rows, weights=weights)
        trials = validate_trials(self._fit_context["trials"], n_rows)
        active = (
            np.ones(n_rows, dtype=bool) if weights is None else np.asarray(weights) > 0
        )
        if not np.any((trials >= 2) & active):
            raise ValueError(
                "betabinomial_lss dispersion is unidentified when all positive-weight rows have fewer than two trials."
            )

    def validate_y(self, y):
        y = super().validate_y(y)
        trials = self.fit_context(len(y))["trials"]
        if np.any((y < 0) | (y > trials) | (y != np.floor(y))):
            raise ValueError(
                "Beta-binomial responses must be integer successes between zero and trials."
            )
        return y

    def _log_density(self, y, eta_mu, log_kappa):
        trials = self.fit_context(len(y))["trials"]
        mu, kappa = 1 / (1 + (-eta_mu).exp()), log_kappa.exp()
        a, b = mu * kappa, (1 - mu) * kappa
        # gamlss.dist/R/BB.R dBB with sigma=1/kappa. Differentiate this
        # normalized likelihood, not BB's squared-score/zero-cross curvature.
        result = (
            gammaln(trials + 1)
            - gammaln(y + 1)
            - gammaln(trials - y + 1)
            + (y + a).lgamma()
            + (trials - y + b).lgamma()
            - (trials + kappa).lgamma()
            + kappa.lgamma()
            - a.lgamma()
            - b.lgamma()
        )
        # A zero-trial row is a point mass at zero, independent of every
        # predictor. Preserve exact zero information despite gamma cancellation.
        if np.any(trials == 0):
            result.coefficients = {
                index: np.where(trials == 0, 0.0, value)
                for index, value in result.coefficients.items()
            }
        return result

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y = self.validate_y(y)
        active = (
            np.ones(len(y), dtype=bool) if weights is None else np.asarray(weights) > 0
        )
        trials = self.fit_context(len(y))["trials"]
        if np.sum(y[active]) == 0 or np.sum((trials - y)[active]) == 0:
            raise ValueError(
                "Beta-binomial fitting requires positive-weight successes and failures; boundary fits are unsupported."
            )
        return super().initialize(y, X, jj, offset=offset, weights=weights, E=E)

    def _initial_predictors(self, y):
        trials = self.fit_context(len(y))["trials"]
        return logit((y + 0.5) / (trials + 1)), 0.0

    def _distribution(self, parameters):
        mu, kappa = self._check_parameters(parameters).T
        return DistributionPrediction(
            BetaBinomialLaw(), {"mu": mu, "kappa": kappa}, self.fit_context(len(mu))
        )

    def logpdf_from_parameters(self, y, parameters):
        return self._distribution(parameters).log_prob(self.validate_y(y))

    def point_prediction_from_parameters(self, parameters):
        return self._distribution(parameters).mean()

    def variance_from_parameters(self, parameters):
        return self._distribution(parameters).variance()

    def rd(self, mu, wt, scale, *, rng):
        return self._distribution(mu).sample(n_draws=1, random_state=rng)[0]
