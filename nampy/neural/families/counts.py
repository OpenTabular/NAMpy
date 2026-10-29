"""Torch distributional families: counts."""

from typing import Callable, Union

import numpy as np
import torch
import torch.distributions as dist

from ..objectives.parameterized import ParameterizedObjective


class PoissonDistribution(ParameterizedObjective):
    """Poisson with parameter [rate]."""

    point_summary = "mean"

    def __init__(
        self,
        name: str = "Poisson",
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["rate"], eps=eps)
        self.rate_transform = rate_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("PoissonDistribution requires non-negative targets.")

        rate = self.get_transform(self.rate_transform)(predictions[:, 0])
        return -dist.Poisson(rate=rate).log_prob(y)

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)
        y = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred)
        rate = np.clip(pred[:, 0], 1e-9, None)
        err = y - rate

        # Safe Poisson deviance
        term = np.where(y > 0, y * np.log(np.clip(y, 1e-12, None) / rate), 0.0)
        poisson_dev = 2.0 * np.sum(term - (y - rate))

        metrics.update(
            {
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "poisson_deviance": float(poisson_dev),
            }
        )
        return metrics

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return torch.sqrt(torch.clamp(params[:, 0], min=0.0))


class NegativeBinomialDistribution(ParameterizedObjective):
    """
    Negative Binomial with parameters [mean, dispersion].

    Parameterization
    ----------------
    We use `dispersion = alpha > 0` such that:
        Var(Y) = mean + alpha * mean^2

    PyTorch `NegativeBinomial` expects `(total_count=r, probs=p)` with mean:
        mean = r * p / (1 - p)
    Therefore:
        r = 1 / alpha
        p = mean / (mean + r)
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "NegativeBinomial",
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["mean", "dispersion"], eps=eps)
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError(
                "NegativeBinomialDistribution requires non-negative targets."
            )

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        alpha = self.get_transform(self.dispersion_transform)(predictions[:, 1])

        total_count = 1.0 / torch.clamp(alpha, min=self.eps)  # r
        probs = mean / torch.clamp(mean + total_count, min=self.eps)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        nb = dist.NegativeBinomial(total_count=total_count, probs=probs)
        return -nb.log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        """NB mean parameter."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        mean, dispersion = params[:, 0], params[:, 1]
        return torch.sqrt(torch.clamp(mean + dispersion * mean.square(), min=0.0))


def _unit_interval_transform(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe sigmoid transform to (0, 1)."""
    return torch.sigmoid(torch.clamp(x, min=-40.0, max=40.0))


def _mean_dispersion_to_nb(mean: torch.Tensor, alpha: torch.Tensor, eps: float):
    """
    Convert NB2 mean/dispersion parameterization to PyTorch's
    NegativeBinomial(total_count=r, probs=p) parameterization.

    NB2:
        Var(Y) = mean + alpha * mean^2
        r = 1 / alpha
        p = mean / (mean + r)
    """
    total_count = 1.0 / torch.clamp(alpha, min=eps)
    probs = mean / torch.clamp(mean + total_count, min=eps)
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    return total_count, probs


class ZeroInflatedPoissonDistribution(ParameterizedObjective):
    """
    Zero-Inflated Poisson with parameters [zero_prob, rate].

    Interpretation
    --------------
    P(Y = 0) = zero_prob + (1 - zero_prob) * Pois(rate).pmf(0)
    P(Y = y>0) = (1 - zero_prob) * Pois(rate).pmf(y)

    Notes
    -----
    * Targets must be non-negative.
    * `predict_point()` returns the mean:
          E[Y] = (1 - zero_prob) * rate
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "ZeroInflatedPoisson",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["zero_prob", "rate"], eps=eps)
        self.zero_prob_transform = zero_prob_transform
        self.rate_transform = rate_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError(
                "ZeroInflatedPoissonDistribution requires non-negative targets."
            )

        zero_prob = self.get_transform(self.zero_prob_transform)(predictions[:, 0])
        zero_prob = torch.clamp(zero_prob, min=self.eps, max=1.0 - self.eps)
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])

        pois = dist.Poisson(rate=rate)
        log_pois = pois.log_prob(y)
        log_pois0 = pois.log_prob(torch.zeros_like(y))

        log_prob_zero = torch.logaddexp(
            torch.log(zero_prob), torch.log1p(-zero_prob) + log_pois0
        )
        log_prob_pos = torch.log1p(-zero_prob) + log_pois

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        rate = params[:, 1]
        return (1.0 - zero_prob) * rate

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob, rate = params[:, 0], params[:, 1]
        variance = (1.0 - zero_prob) * rate * (1.0 + zero_prob * rate)
        return torch.sqrt(torch.clamp(variance, min=0.0))


class ZeroInflatedNegativeBinomialDistribution(ParameterizedObjective):
    """
    Zero-Inflated Negative Binomial (NB2) with parameters [zero_prob, mean, dispersion].

    NB2 parameterization:
        Var(Y) = mean + dispersion * mean^2

    Notes
    -----
    * Targets must be non-negative.
    * `predict_point()` returns the mean:
          E[Y] = (1 - zero_prob) * mean
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "ZeroInflatedNegativeBinomial",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(
            name=name, param_names=["zero_prob", "mean", "dispersion"], eps=eps
        )
        self.zero_prob_transform = zero_prob_transform
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError(
                "ZeroInflatedNegativeBinomialDistribution requires non-negative targets."
            )

        zero_prob = self.get_transform(self.zero_prob_transform)(predictions[:, 0])
        zero_prob = torch.clamp(zero_prob, min=self.eps, max=1.0 - self.eps)
        mean = self.get_transform(self.mean_transform)(predictions[:, 1])
        dispersion = self.get_transform(self.dispersion_transform)(predictions[:, 2])

        total_count, probs = _mean_dispersion_to_nb(mean, dispersion, self.eps)
        nb = dist.NegativeBinomial(total_count=total_count, probs=probs)

        log_nb = nb.log_prob(y)
        log_nb0 = nb.log_prob(torch.zeros_like(y))

        log_prob_zero = torch.logaddexp(
            torch.log(zero_prob), torch.log1p(-zero_prob) + log_nb0
        )
        log_prob_pos = torch.log1p(-zero_prob) + log_nb

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        mean = params[:, 1]
        return (1.0 - zero_prob) * mean

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob, mean, dispersion = params[:, 0], params[:, 1], params[:, 2]
        base_variance = mean + dispersion * mean.square()
        variance = (1.0 - zero_prob) * (base_variance + zero_prob * mean.square())
        return torch.sqrt(torch.clamp(variance, min=0.0))


class HurdlePoissonDistribution(ParameterizedObjective):
    """
    Hurdle Poisson with parameters [zero_prob, rate].

    Interpretation
    --------------
    P(Y = 0) = zero_prob
    P(Y = y>0) = (1 - zero_prob) * Pois(rate).pmf(y) / (1 - Pois(rate).pmf(0))

    Notes
    -----
    * Targets must be non-negative.
    * `predict_point()` returns the mean:
          E[Y] = (1 - zero_prob) * E[Poisson(rate) | Y > 0]
               = (1 - zero_prob) * rate / (1 - exp(-rate))
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "HurdlePoisson",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["zero_prob", "rate"], eps=eps)
        self.zero_prob_transform = zero_prob_transform
        self.rate_transform = rate_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("HurdlePoissonDistribution requires non-negative targets.")

        zero_prob = self.get_transform(self.zero_prob_transform)(predictions[:, 0])
        zero_prob = torch.clamp(zero_prob, min=self.eps, max=1.0 - self.eps)
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])

        pois = dist.Poisson(rate=rate)
        log_pois = pois.log_prob(y)
        log_pois0 = pois.log_prob(torch.zeros_like(y))

        log_trunc_norm = torch.log(
            torch.clamp(1.0 - torch.exp(log_pois0), min=self.eps)
        )
        log_prob_zero = torch.log(zero_prob)
        log_prob_pos = torch.log1p(-zero_prob) + log_pois - log_trunc_norm

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        rate = params[:, 1]
        trunc_mean = rate / torch.clamp(1.0 - torch.exp(-rate), min=self.eps)
        return (1.0 - zero_prob) * trunc_mean


class HurdleNegativeBinomialDistribution(ParameterizedObjective):
    """
    Hurdle Negative Binomial (NB2) with parameters [zero_prob, mean, dispersion].

    Interpretation
    --------------
    P(Y = 0) = zero_prob
    P(Y = y>0) = (1 - zero_prob) * NB(mean, dispersion).pmf(y) / (1 - NB(...).pmf(0))

    Notes
    -----
    * Targets must be non-negative.
    * `predict_point()` returns the mean:
          E[Y] = (1 - zero_prob) * E[NB(mean, dispersion) | Y > 0]
               = (1 - zero_prob) * mean / (1 - P_NB(Y=0))
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "HurdleNegativeBinomial",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(
            name=name, param_names=["zero_prob", "mean", "dispersion"], eps=eps
        )
        self.zero_prob_transform = zero_prob_transform
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError(
                "HurdleNegativeBinomialDistribution requires non-negative targets."
            )

        zero_prob = self.get_transform(self.zero_prob_transform)(predictions[:, 0])
        zero_prob = torch.clamp(zero_prob, min=self.eps, max=1.0 - self.eps)
        mean = self.get_transform(self.mean_transform)(predictions[:, 1])
        dispersion = self.get_transform(self.dispersion_transform)(predictions[:, 2])

        total_count, probs = _mean_dispersion_to_nb(mean, dispersion, self.eps)
        nb = dist.NegativeBinomial(total_count=total_count, probs=probs)

        log_nb = nb.log_prob(y)
        log_nb0 = nb.log_prob(torch.zeros_like(y))

        log_trunc_norm = torch.log(torch.clamp(1.0 - torch.exp(log_nb0), min=self.eps))
        log_prob_zero = torch.log(zero_prob)
        log_prob_pos = torch.log1p(-zero_prob) + log_nb - log_trunc_norm

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)

        zero_prob = params[:, 0]
        mean = params[:, 1]
        dispersion = params[:, 2]

        total_count, probs = _mean_dispersion_to_nb(mean, dispersion, self.eps)
        nb = dist.NegativeBinomial(total_count=total_count, probs=probs)
        p0 = torch.exp(nb.log_prob(torch.zeros_like(mean)))

        trunc_mean = mean / torch.clamp(1.0 - p0, min=self.eps)
        return (1.0 - zero_prob) * trunc_mean


class TweedieDistribution(ParameterizedObjective):
    """
    Proper Tweedie likelihood for the compound Poisson-Gamma case: 1 < p < 2.

    Parameterization
    ----------------
    We use:
        E[Y] = mean = mu > 0
        Var(Y) = dispersion * mu^p,   with 1 < p < 2

    The model predicts raw outputs for [mean, dispersion], and `variance_power`
    (often called `p`) is fixed at construction time.

    Notes
    -----
    * Supports y >= 0 with exact point mass at zero.
    * For y > 0, the density is evaluated via the infinite series representation
      of the compound Poisson-Gamma distribution, truncated at `series_max_terms`.
    * `predict_point()` returns the mean parameter.
    """

    point_summary = "mean"

    def __init__(
        self,
        variance_power: float = 1.5,
        name: str = "Tweedie",
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        series_max_terms: int = 200,
        eps: float = 1e-8,
    ):
        p = float(variance_power)
        if not (1.0 < p < 2.0):
            raise ValueError(
                "TweedieDistribution currently supports only 1 < variance_power < 2 "
                "(compound Poisson-Gamma case)."
            )
        if int(series_max_terms) < 1:
            raise ValueError("series_max_terms must be >= 1.")

        super().__init__(name=name, param_names=["mean", "dispersion"], eps=eps)
        self.variance_power = p
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform
        self.series_max_terms = int(series_max_terms)

    def _compound_params(self, mean: torch.Tensor, dispersion: torch.Tensor):
        """
        Compound Poisson-Gamma representation for 1 < p < 2.

        If Y = sum_{i=1}^N X_i where:
          N ~ Poisson(lambda)
          X_i ~ Gamma(shape=a, scale=b)
        then:
          lambda = mean^(2-p) / (dispersion * (2-p))
          a      = (2-p) / (p-1)
          b      = dispersion * (p-1) * mean^(p-1)
        """
        p = self.variance_power
        lam = mean.pow(2.0 - p) / torch.clamp(dispersion * (2.0 - p), min=self.eps)
        a = (2.0 - p) / (p - 1.0)
        b = dispersion * (p - 1.0) * mean.pow(p - 1.0)
        b = torch.clamp(b, min=self.eps)
        lam = torch.clamp(lam, min=self.eps)
        return lam, a, b

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("TweedieDistribution requires non-negative targets.")

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        dispersion = self.get_transform(self.dispersion_transform)(predictions[:, 1])

        lam, a, b = self._compound_params(mean, dispersion)

        log_prob = torch.empty_like(y)

        # Exact mass at zero
        zero_mask = y == 0
        if zero_mask.any():
            log_prob[zero_mask] = -lam[zero_mask]

        # Continuous density for y > 0 via series expansion
        pos_mask = ~zero_mask
        if pos_mask.any():
            y_pos = torch.clamp(y[pos_mask], min=self.eps)
            lam_pos = lam[pos_mask]
            b_pos = b[pos_mask]

            n = torch.arange(
                1,
                self.series_max_terms + 1,
                device=y_pos.device,
                dtype=y_pos.dtype,
            ).view(1, -1)

            log_lambda = torch.log(lam_pos).unsqueeze(1)
            log_y = torch.log(y_pos).unsqueeze(1)
            log_b = torch.log(b_pos).unsqueeze(1)

            # Gamma shape for the sum of n jumps
            an = a * n

            # log term_n:
            #   n log(lambda) - lgamma(n+1) - lgamma(a n)
            #   + a n (log y - log b)
            log_terms = (
                n * log_lambda
                - torch.lgamma(n + 1.0)
                - torch.lgamma(an)
                + an * (log_y - log_b)
            )

            log_sum = torch.logsumexp(log_terms, dim=1)

            # f(y) = exp(-lambda - y/b) * (1/y) * sum_n term_n
            log_prob[pos_mask] = -lam_pos - y_pos / b_pos - torch.log(y_pos) + log_sum

        return -log_prob

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        mean, dispersion = params[:, 0], params[:, 1]
        variance = dispersion * mean.pow(self.variance_power)
        return torch.sqrt(torch.clamp(variance, min=0.0))
