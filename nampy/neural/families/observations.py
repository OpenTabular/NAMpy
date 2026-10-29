"""Continuous censored/truncated neural likelihoods with shared metadata."""

from collections.abc import Callable

import numpy as np
import torch

from ...distributions import (
    DistributionLaw,
    DistributionPrediction,
    LogNormalLaw,
    NormalLaw,
    ObservationTarget,
    TruncatedLaw,
    WeibullLaw,
)
from ..objectives.parameterized import ParameterizedObjective
from .gamlss import _raw


def normal_interval_log_mass(lower, upper, mean, log_scale):
    result = torch.zeros_like(mean)
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    if torch.any(finite):
        a = (lower[finite] - mean[finite]) * torch.exp(-log_scale[finite])
        b = (upper[finite] - mean[finite]) * torch.exp(-log_scale[finite])
        use_cdf = (lower[finite] + upper[finite]) / 2 <= mean[finite]
        low = torch.where(
            use_cdf, torch.special.log_ndtr(a), torch.special.log_ndtr(-b)
        )
        high = torch.where(
            use_cdf, torch.special.log_ndtr(b), torch.special.log_ndtr(-a)
        )
        result[finite] = high + torch.log(-torch.expm1(low - high))
    left = torch.isneginf(lower) & torch.isfinite(upper)
    result[left] = torch.special.log_ndtr(
        (upper[left] - mean[left]) * torch.exp(-log_scale[left])
    )
    right = torch.isfinite(lower) & torch.isposinf(upper)
    result[right] = torch.special.log_ndtr(
        (mean[right] - lower[right]) * torch.exp(-log_scale[right])
    )
    return result


class _ObservationDistribution(ParameterizedObjective):
    """Compose continuous density/interval kernels with fixed truncation.

    Model parameters describe the latent normal. Prediction laws condition on
    truncation only; they do not condition on a particular censoring observation.
    """

    point_summary = "median"
    base_law: Callable[[], DistributionLaw] = NormalLaw
    _exact_log_density: Callable
    _interval_mass: Callable
    parameter_names = ("mean", "scale")
    positive_parameters = (False, True)
    support_lower = -np.inf
    family_name = "NormalObservations"
    supports_observation_target = True
    target_dtype = torch.float64

    def __init__(self, *, lower=-np.inf, upper=np.inf):
        super().__init__(self.family_name, list(self.parameter_names))
        self.law = TruncatedLaw(self.base_law(), max(lower, self.support_lower), upper)

    def encode_targets(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        target = (
            value
            if isinstance(value, ObservationTarget)
            else ObservationTarget.from_encoded(value)
        )
        exact = target.lower == target.upper
        if np.any(
            exact
            & ((target.lower <= self.law.lower) | (target.upper >= self.law.upper))
        ):
            raise ValueError("Exact observations must be inside truncation bounds.")
        if np.any(
            ~exact
            & (
                np.maximum(target.lower, self.law.lower)
                >= np.minimum(target.upper, self.law.upper)
            )
        ):
            raise ValueError("Censoring intervals must overlap the truncated support.")
        return target.encoded

    def preprocessing_targets(self, targets):
        return self.encode_targets(targets)[:, 0]

    def forward(self, predictions):
        p = _raw(predictions, 2)
        return torch.stack(
            [
                p[:, i].exp() if positive else p[:, i]
                for i, positive in enumerate(self.positive_parameters)
            ],
            dim=1,
        )

    def compute_loss_per_sample(self, predictions, y_true):
        p = _raw(predictions, 2)
        if isinstance(y_true, ObservationTarget):
            y_true = self.encode_targets(y_true)
        encoded = torch.as_tensor(y_true, device=p.device, dtype=p.dtype)
        if encoded.shape != (len(p), 3) or not torch.isfinite(encoded).all():
            raise ValueError("Observation targets require finite aligned encoded rows.")
        yy, endpoint, code = encoded.T
        if not torch.isin(
            code, torch.tensor([-1, 0, 1, 2, 3], device=p.device, dtype=p.dtype)
        ).all():
            raise ValueError("Unknown observation code.")
        exact = code == 0
        lower = torch.where((code == -1) | (code == 3), -torch.inf, yy)
        upper = torch.where((code == 1) | (code == 3), torch.inf, endpoint)
        if torch.any(
            exact & ((yy != endpoint) | (yy <= self.law.lower) | (yy >= self.law.upper))
        ):
            raise ValueError("Exact observations must be inside truncation bounds.")
        lo = torch.clamp(lower, min=self.law.lower)
        hi = torch.clamp(upper, max=self.law.upper)
        if torch.any((~exact) & (lo >= hi)):
            raise ValueError("Censoring intervals must overlap truncated support.")
        mean, log_scale = p.T
        value = torch.zeros_like(mean)
        value[exact] = self._exact_log_density(yy[exact], mean[exact], log_scale[exact])
        value[~exact] = self._interval_mass(
            lo[~exact], hi[~exact], mean[~exact], log_scale[~exact]
        )
        lower, upper = (
            torch.full_like(mean, self.law.lower),
            torch.full_like(mean, self.law.upper),
        )
        return self._interval_mass(lower, upper, mean, log_scale) - value

    def distribution_prediction(self, parameters):
        return DistributionPrediction(
            self.law,
            dict(
                zip(
                    self.law.base.spec.parameter_names,
                    np.asarray(parameters).T,
                    strict=True,
                )
            ),
        )

    def _moment(self, predictions, transformed, spread):
        p = predictions if transformed else self(predictions)
        law = self.distribution_prediction(p.detach().cpu().numpy())
        value = (
            np.sqrt(law.variance())
            if spread
            else (
                law.mean() if self.point_summary == "mean" else law.quantile(0.5)[:, 0]
            )
        )
        return torch.as_tensor(value, device=p.device, dtype=p.dtype)

    def predict_point(self, predictions, transformed=False):
        return self._moment(predictions, transformed, False)

    def predict_spread(self, predictions, transformed=False):
        return self._moment(predictions, transformed, True)


class NormalObservationsDistribution(_ObservationDistribution):
    """Normal location/SD with explicit censoring and fixed truncation."""

    point_summary = "mean"
    _interval_mass = staticmethod(normal_interval_log_mass)

    @staticmethod
    def _exact_log_density(y, mean, log_scale):
        return (
            -0.5 * np.log(2 * np.pi)
            - log_scale
            - 0.5 * ((y - mean) * torch.exp(-log_scale)) ** 2
        )


class LogNormalObservationsDistribution(_ObservationDistribution):
    """Lognormal log-location/log-SD; conditional median point prediction."""

    family_name = "LogNormalObservations"
    base_law = LogNormalLaw
    parameter_names = ("loc", "scale")
    support_lower = 0.0

    @staticmethod
    def _exact_log_density(y, loc, log_scale):
        return (
            NormalObservationsDistribution._exact_log_density(y.log(), loc, log_scale)
            - y.log()
        )

    @staticmethod
    def _interval_mass(lower, upper, loc, log_scale):
        return normal_interval_log_mass(lower.log(), upper.log(), loc, log_scale)


class WeibullObservationsDistribution(_ObservationDistribution):
    """Weibull scale/shape; conditional median point prediction."""

    family_name = "WeibullObservations"
    base_law = WeibullLaw
    parameter_names = ("scale", "shape")
    positive_parameters = (True, True)
    support_lower = 0.0

    @staticmethod
    def _exact_log_density(y, log_scale, log_shape):
        t = log_shape.exp() * (y.log() - log_scale)
        return log_shape - y.log() + t - t.exp()

    @staticmethod
    def _interval_mass(lower, upper, log_scale, log_shape):
        lo = torch.zeros_like(log_scale)
        active = lower > 0
        lo[active] = -(
            log_shape[active].exp() * (lower[active].log() - log_scale[active])
        ).exp()
        value = lo.clone()
        finite = torch.isfinite(upper)
        hi = -(
            log_shape[finite].exp() * (upper[finite].log() - log_scale[finite])
        ).exp()
        value[finite] = lo[finite] + torch.log(-torch.expm1(hi - lo[finite]))
        return value
