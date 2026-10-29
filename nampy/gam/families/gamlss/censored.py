"""Censored survival adapters with fourth-order derivatives and joint REML.

Reuses the generic mgcv joint gradient/Hessian assembly. Censoring and
entry are per-fit observation context, never observation weights.
"""

import numpy as np

from ....survival.contracts import SurvivalTarget
from ....survival.likelihood import log_entry_ratio, validate_parametric_fit
from ._taylor import Taylor
from .positive import (
    LogNormalLocationScale,
    WeibullLocationShape,
    _PositiveDistributionalFamily,
)


class _SurvivalObservation(_PositiveDistributionalFamily):
    context_names = ("event", "entry")
    max_derivative_order = 4
    supports_reml = True
    supports_ml = supports_laml = False
    supported_smoothing_optimizers = frozenset({"outer_newton", "bfgs"})

    def bind_fit_context(self, context, n_rows, *, weights=None):
        super().bind_fit_context(context, n_rows, weights=weights)
        event, entry = (self._fit_context[name] for name in self.context_names)
        if not np.isin(event, [0, 1]).all() or np.any(entry < 0):
            raise ValueError("Survival context needs event 0/1 and nonnegative entry.")

    def _target(self, y):
        context = self.fit_context(len(y))
        return SurvivalTarget(y, context["event"], context["entry"])

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        target = self._target(y)
        weights = validate_parametric_fit(target, weights)
        event = target.event & (weights > 0)
        log_t = np.log(target.time[event])
        event_weights = weights[event]
        if self.name == "censored_weibull":
            # Exact exponential submodel MLE with entry, not an uncensored fit.
            log_scale = np.log(
                np.sum(weights * (target.time - target.entry))
                / np.sum(weights * target.event)
            )
            starts = (log_scale, 0.0)
        else:
            loc = np.average(log_t, weights=event_weights)
            sd = np.sqrt(np.average((log_t - loc) ** 2, weights=event_weights))
            starts = (loc, np.log(max(sd, 0.1)))
        # The generic initializer owns design projection, offsets and QR.
        self._survival_starts = starts
        try:
            return super().initialize(y, X, jj, offset=offset, weights=weights, E=E)
        finally:
            del self._survival_starts

    def _initial_predictors(self, y):
        return self._survival_starts

    def residuals(self, *args, **kwargs):
        raise NotImplementedError(
            "Uncensored response/Pearson residuals are unavailable for censored observations."
        )


class CensoredWeibull(_SurvivalObservation, WeibullLocationShape):
    """Right censoring/entry; log scale/log shape; fixed and Newton/BFGS REML."""

    name = "censored_weibull"

    def _log_density(self, y, log_scale, log_shape):
        target = self._target(y)
        shape = log_shape.exp()
        u = shape * (np.log(y) - log_scale)
        positive = target.entry > 0
        ratio = log_entry_ratio(y[positive], target.entry[positive])
        fraction = Taylor.merge(
            positive,
            -(shape.subset(positive) * ratio).expm1(),
            shape.subset(~positive).constant(np.ones(np.sum(~positive))),
        )
        return target.event * (log_shape - np.log(y) + u) - u.exp() * fraction


class CensoredLogNormal(_SurvivalObservation, LogNormalLocationScale):
    """Right censoring/entry; location/log SD; fixed and Newton/BFGS REML."""

    name = "censored_lognormal"

    def _log_density(self, y, loc, log_scale):
        target = self._target(y)
        event = target.event
        z = (np.log(y) - loc) * (-log_scale).exp()
        # Evaluate each observation branch only on its own rows.
        density = (
            -0.5 * np.log(2 * np.pi)
            - log_scale.subset(event)
            - 0.5 * z.subset(event) ** 2
            - np.log(y[event])
        )
        value = Taylor.merge(event, density, z.subset(~event).log_normal_survival())
        positive = target.entry > 0
        za = (np.log(target.entry[positive]) - loc.subset(positive)) * (
            -log_scale.subset(positive)
        ).exp()
        normalizer = Taylor.merge(
            positive,
            za.log_normal_survival(),
            loc.subset(~positive).constant(np.zeros(np.sum(~positive))),
        )
        return value - normalizer
