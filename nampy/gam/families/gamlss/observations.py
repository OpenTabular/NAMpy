"""Continuous observation likelihoods through the existing joint Taylor adapter.

Normal/lognormal kernels follow gamlss.dist/R/logNO.R; Weibull follows
R/WEI.R. Censoring contributes interval probability and truncation its
normalizing probability. These are NAMpy observation extensions, not cnorm.

References
----------
- Lawless, J. F. (2003). Statistical Models and Methods for Lifetime Data.
  https://doi.org/10.1002/9781118033005
- Rigby, R. A., and Stasinopoulos, D. M. (2005). Generalized additive models for
  location, scale and shape.
  https://doi.org/10.1111/j.1467-9876.2005.00510.x
"""

from collections.abc import Callable

import numpy as np

from ....distributions import (
    DistributionLaw,
    DistributionPrediction,
    LogNormalLaw,
    NormalLaw,
    WeibullLaw,
)
from ....distributions.observations import ObservationTarget, TruncatedLaw
from ._taylor import Taylor
from .distributional import _DistributionalFamily


def normal_interval_jet(lower, upper, loc, log_scale):
    """Analytic normal interval log mass through the Taylor engine's order."""
    result = loc.constant(np.zeros(len(lower)))
    finite = np.isfinite(lower) & np.isfinite(upper)
    if finite.any():
        mu, ls = loc.subset(finite), log_scale.subset(finite)
        a, b = (lower[finite] - mu) * (-ls).exp(), (upper[finite] - mu) * (-ls).exp()
        use_cdf = (lower[finite] + upper[finite]) / 2 <= mu.value
        low = Taylor.merge(
            use_cdf,
            (-a.subset(use_cdf)).log_normal_survival(),
            b.subset(~use_cdf).log_normal_survival(),
        )
        high = Taylor.merge(
            use_cdf,
            (-b.subset(use_cdf)).log_normal_survival(),
            a.subset(~use_cdf).log_normal_survival(),
        )
        value = high + (-(low - high).expm1()).log()
        result = Taylor.merge(finite, value, result.subset(~finite))
    left = np.isneginf(lower) & np.isfinite(upper)
    if left.any():
        value = -((upper[left] - loc.subset(left)) * (-log_scale.subset(left)).exp())
        result = Taylor.merge(left, value.log_normal_survival(), result.subset(~left))
    right = np.isfinite(lower) & np.isposinf(upper)
    if right.any():
        value = (lower[right] - loc.subset(right)) * (-log_scale.subset(right)).exp()
        result = Taylor.merge(right, value.log_normal_survival(), result.subset(~right))
    return result


class _ObservationFamily(_DistributionalFamily):
    """Compose continuous density/interval kernels with fixed truncation.

    ObservationTarget supplies explicit per-fit metadata. Independent rows,
    fixed smoothing and exact Newton/BFGS REML are supported; variable truncation
    limits, discrete/mixed laws and saturated/deviance residuals are unsupported.
    This family does not alter mgcv's source-specific cnorm behavior.
    """

    point_summary = "median"
    base_law: Callable[[], DistributionLaw] = NormalLaw
    _exact_log_density: Callable
    _interval_mass: Callable
    support_lower = -np.inf
    name = "normal_observation_lss"
    parameter_names = ("mean", "scale")
    link_names = ("identity", "log")
    parameter_bounds = {"scale": (0, None)}
    context_names = ("observation_upper", "observation_code")
    supports_observation_target = True
    prediction_context_names = ()
    max_derivative_order = 4
    supports_reml = True
    supported_smoothing_optimizers = frozenset({"outer_newton", "bfgs"})

    def __init__(self, *, lower=-np.inf, upper=np.inf):
        self.law = TruncatedLaw(self.base_law(), max(lower, self.support_lower), upper)
        super().__init__()

    def _target(self, y):
        context = self.fit_context(len(y))
        return ObservationTarget.from_encoded(
            np.column_stack(
                [y, context["observation_upper"], context["observation_code"]]
            )
        )

    def validate_y(self, y):
        y = super().validate_y(y)
        target = self._target(y)
        exact = target.lower == target.upper
        if np.any(exact & ((y <= self.law.lower) | (y >= self.law.upper))):
            raise ValueError("Exact observations must be inside truncation bounds.")
        if np.any(
            ~exact
            & (
                np.maximum(target.lower, self.law.lower)
                >= np.minimum(target.upper, self.law.upper)
            )
        ):
            raise ValueError("Censoring intervals must overlap the truncated support.")
        return y

    def _log_density(self, y, loc, log_scale):
        target = self._target(y)
        exact = target.lower == target.upper
        density = self._exact_log_density(
            y[exact], loc.subset(exact), log_scale.subset(exact)
        )
        lower = np.maximum(target.lower[~exact], self.law.lower)
        upper = np.minimum(target.upper[~exact], self.law.upper)
        mass = self._interval_mass(
            lower, upper, loc.subset(~exact), log_scale.subset(~exact)
        )
        numerator = Taylor.merge(exact, density, mass)
        denominator = self._interval_mass(
            np.full(len(y), self.law.lower),
            np.full(len(y), self.law.upper),
            loc,
            log_scale,
        )
        return numerator - denominator

    def residuals(self, *args, **kwargs):
        raise NotImplementedError(
            "Ordinary response/Pearson/deviance residuals are unavailable for censored observations; use explicit observation likelihood evaluation."
        )

    def _initial_predictors(self, y):
        sd = np.std(y)
        if not np.isfinite(sd) or sd == 0:
            raise NotImplementedError(
                "Identical observation thresholds have no supported location-scale initialization."
            )
        return np.mean(y), np.log(sd)

    def _distribution(self, parameters):
        p = self._check_parameters(parameters)
        return DistributionPrediction(
            self.law, dict(zip(self.law.base.spec.parameter_names, p.T, strict=True))
        )

    def logpdf_from_parameters(self, y, parameters):
        from ....distributions.observations import observation_log_prob

        return observation_log_prob(self._distribution(parameters), self._target(y))

    def point_prediction_from_parameters(self, parameters):
        return self._distribution(parameters).quantile(0.5)[:, 0]

    def variance_from_parameters(self, parameters):
        return self._distribution(parameters).variance()


class NormalObservations(_ObservationFamily):
    """Normal location/SD with explicit censoring and fixed truncation.

    Predictions summarize the conditional mean; fourth-order analytic
    derivatives support fixed and automatic smoothing in the joint engine.
    """

    point_summary = "mean"
    _interval_mass = staticmethod(normal_interval_jet)

    @staticmethod
    def _exact_log_density(y, loc, log_scale):
        z = (y - loc) * (-log_scale).exp()
        return -0.5 * np.log(2 * np.pi) - log_scale - 0.5 * z * z

    def point_prediction_from_parameters(self, parameters):
        return self._distribution(parameters).mean()


class LogNormalObservations(_ObservationFamily):
    """Lognormal log-location/log-SD, censoring and fixed truncation.

    Natural parameters are (loc, scale); the default point is the conditional
    median. Conditional response moments are unavailable in the generic law.
    """

    name = "lognormal_observation_lss"
    parameter_names = ("loc", "scale")
    base_law = LogNormalLaw
    support_lower = 0.0

    @staticmethod
    def _exact_log_density(y, loc, log_scale):
        return NormalObservations._exact_log_density(
            np.log(y), loc, log_scale
        ) - np.log(y)

    @staticmethod
    def _interval_mass(lower, upper, loc, log_scale):
        with np.errstate(divide="ignore"):
            return normal_interval_jet(np.log(lower), np.log(upper), loc, log_scale)

    def _initial_predictors(self, y):
        return super()._initial_predictors(np.log(y[y > 0]))


class WeibullObservations(_ObservationFamily):
    """Weibull scale/shape, censoring and fixed truncation; conditional median.

    Both predictors use log links. Analytic likelihood derivatives through four
    support the existing Newton/BFGS joint REML engine.
    """

    name = "weibull_observation_lss"
    parameter_names = ("scale", "shape")
    link_names = ("log", "log")
    parameter_bounds = {"scale": (0, None), "shape": (0, None)}
    base_law = WeibullLaw
    support_lower = 0.0

    @staticmethod
    def _exact_log_density(y, log_scale, log_shape):
        t = log_shape.exp() * (np.log(y) - log_scale)
        return log_shape - np.log(y) + t - t.exp()

    @staticmethod
    def _interval_mass(lower, upper, log_scale, log_shape):
        # WEI survival is exp(-(y/scale)**shape). Compute differences in
        # survival space to retain right-tail interval precision.
        lo = log_scale.constant(np.zeros(len(lower)))
        active = lower > 0
        if active.any():
            t = log_shape.subset(active).exp() * (
                np.log(lower[active]) - log_scale.subset(active)
            )
            lo = Taylor.merge(active, -t.exp(), lo.subset(~active))
        value = lo
        finite = np.isfinite(upper)
        if finite.any():
            t = log_shape.subset(finite).exp() * (
                np.log(upper[finite]) - log_scale.subset(finite)
            )
            hi = -t.exp()
            mass = lo.subset(finite) + (-(hi - lo.subset(finite)).expm1()).log()
            value = Taylor.merge(finite, mass, value.subset(~finite))
        return value

    def _initial_predictors(self, y):
        log_y = np.log(y[y > 0])
        if not len(log_y) or np.std(log_y) == 0:
            raise ValueError(
                "Weibull observations require distinct positive thresholds."
            )
        shape = np.pi / (np.sqrt(6) * np.std(log_y))
        return np.mean(log_y) + np.euler_gamma / shape, np.log(shape)
