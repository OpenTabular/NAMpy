"""Torch distributional families: continuous."""

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F

from ..objectives.parameterized import ParameterizedObjective, TensorLike


class NormalDistribution(ParameterizedObjective):
    """
    Gaussian with parameters [mean, scale].

    Notes
    -----
    `scale` is the standard deviation (NOT the variance).
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "Normal",
        mean_transform: Union[str, Callable] = "none",
        scale_transform: Union[str, Callable] = "positive",
        eps: float = 1e-6,
    ):
        super().__init__(name=name, param_names=["mean", "scale"], eps=eps)
        self.mean_transform = mean_transform
        self.scale_transform = scale_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        scale = self.get_transform(self.scale_transform)(predictions[:, 1])
        return -dist.Normal(loc=mean, scale=scale).log_prob(y)

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)
        y = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred)
        mu = pred[:, 0]
        err = y - mu
        metrics.update(
            {
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
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
        return params[:, 1]


class InverseGammaDistribution(ParameterizedObjective):
    """Inverse-Gamma with parameters [shape, rate]."""

    point_summary = "mean"

    def __init__(
        self,
        name: str = "InverseGamma",
        shape_transform: Union[str, Callable] = "positive",
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["shape", "rate"], eps=eps)
        self.shape_transform = shape_transform
        self.rate_transform = rate_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError(
                "InverseGammaDistribution requires strictly positive targets."
            )

        shape = self.get_transform(self.shape_transform)(predictions[:, 0])
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])
        return -dist.InverseGamma(concentration=shape, rate=rate).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        """
        Mean exists for shape > 1 and equals rate / (shape - 1).
        Returns a numerically safe approximation.
        """
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        shape = params[:, 0]
        rate = params[:, 1]
        return rate / torch.clamp(shape - 1.0, min=self.eps)

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        shape, rate = params[:, 0], params[:, 1]
        variance = rate.square() / (
            (shape - 1.0).square() * torch.clamp(shape - 2.0, min=self.eps)
        )
        return torch.where(
            shape > 2.0,
            torch.sqrt(torch.clamp(variance, min=0.0)),
            torch.full_like(shape, torch.nan),
        )


class BetaDistribution(ParameterizedObjective):
    """Beta with parameters [alpha, beta]. Targets must lie in (0, 1)."""

    point_summary = "mean"

    def __init__(
        self,
        name: str = "Beta",
        shape_transform: Union[str, Callable] = "positive",
        scale_transform: Union[str, Callable] = "positive",
        target_eps: float = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["alpha", "beta"], eps=eps)
        self.alpha_transform = shape_transform
        self.beta_transform = scale_transform
        self.target_eps = float(target_eps)

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        y = torch.clamp(y, min=self.target_eps, max=1.0 - self.target_eps)
        alpha = self.get_transform(self.alpha_transform)(predictions[:, 0])
        beta = self.get_transform(self.beta_transform)(predictions[:, 1])
        return -dist.Beta(concentration1=alpha, concentration0=beta).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        """Beta mean = alpha / (alpha + beta)."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        alpha = params[:, 0]
        beta = params[:, 1]
        return alpha / torch.clamp(alpha + beta, min=self.eps)

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        alpha, beta = params[:, 0], params[:, 1]
        total = torch.clamp(alpha + beta, min=self.eps)
        variance = alpha * beta / (total.square() * (total + 1.0))
        return torch.sqrt(torch.clamp(variance, min=0.0))


class GammaDistribution(ParameterizedObjective):
    """Gamma with parameters [shape, rate]. Targets must be > 0."""

    point_summary = "mean"

    def __init__(
        self,
        name: str = "Gamma",
        shape_transform: Union[str, Callable] = "positive",
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["shape", "rate"], eps=eps)
        self.shape_transform = shape_transform
        self.rate_transform = rate_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("GammaDistribution requires strictly positive targets.")

        shape = self.get_transform(self.shape_transform)(predictions[:, 0])
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])
        return -dist.Gamma(concentration=shape, rate=rate).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        """Gamma mean = shape / rate."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0] / torch.clamp(params[:, 1], min=self.eps)

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return torch.sqrt(params[:, 0]) / torch.clamp(params[:, 1], min=self.eps)


class StudentTDistribution(ParameterizedObjective):
    """Student-t with parameters [df, loc, scale]."""

    point_summary = "location"

    def __init__(
        self,
        name: str = "StudentT",
        df_transform: Union[str, Callable] = "positive",
        loc_transform: Union[str, Callable] = "none",
        scale_transform: Union[str, Callable] = "positive",
        min_df: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["df", "loc", "scale"], eps=eps)
        self.df_transform = df_transform
        self.loc_transform = loc_transform
        self.scale_transform = scale_transform
        self.min_df = float(min_df)

    def forward(self, predictions: TensorLike) -> torch.Tensor:
        params = super().forward(predictions)
        params = params.clone()
        params[:, 0] = params[:, 0] + self.min_df
        return params

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        params = self.forward(predictions)
        df, loc, scale = params[:, 0], params[:, 1], params[:, 2]
        return -dist.StudentT(df=df, loc=loc, scale=scale).log_prob(y)

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)
        y = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred)
        loc = pred[:, 1]
        err = y - loc
        metrics.update(
            {
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )
        return metrics

    def predict_point(self, predictions, transformed: bool = False):
        """Student-t location parameter."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 1]

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        df, scale = params[:, 0], params[:, 2]
        spread = scale * torch.sqrt(df / torch.clamp(df - 2.0, min=self.eps))
        return torch.where(df > 2.0, spread, torch.full_like(spread, torch.nan))


class RobustNormalDistribution(ParameterizedObjective):
    """
    Robustified Gaussian likelihood with parameters [mean, scale].

    The robustness transformation is:
        loglik_rob = log((1 + exp(loglik + rob)) / (1 + exp(rob)))
    This downweights very unlikely observations while remaining smooth.
    """

    point_summary = "location"

    def __init__(
        self,
        name: str = "RobustNormal",
        mean_transform: Union[str, Callable] = "none",
        scale_transform: Union[str, Callable] = "positive",
        rob: Optional[float] = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__(name=name, param_names=["mean", "scale"], eps=eps)
        self.mean_transform = mean_transform
        self.scale_transform = scale_transform
        self.rob = None if rob is None else float(rob)

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        scale = self.get_transform(self.scale_transform)(predictions[:, 1])

        normal_dist = dist.Normal(loc=mean, scale=scale)
        log_likelihood = normal_dist.log_prob(y)

        if self.rob is not None:
            # numerically stable implementation using logaddexp:
            # log(1 + exp(a)) = logaddexp(0, a)
            rob_t = torch.tensor(
                self.rob, device=log_likelihood.device, dtype=log_likelihood.dtype
            )
            log_num = torch.logaddexp(
                torch.zeros_like(log_likelihood), log_likelihood + rob_t
            )
            log_den = torch.logaddexp(
                torch.tensor(0.0, device=rob_t.device, dtype=rob_t.dtype), rob_t
            )
            log_likelihood = log_num - log_den

        return -log_likelihood

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)
        y = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred)
        mu = pred[:, 0]
        err = y - mu
        metrics.update(
            {
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )
        return metrics

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]


class LogNormalDistribution(ParameterizedObjective):
    """
    Log-Normal with parameters [loc, scale].

    Here `loc` and `scale` are the parameters of log(Y):
        log(Y) ~ Normal(loc, scale)

    Notes
    -----
    * Targets must be strictly positive.
    * `predict_point()` returns the conditional mean:
          E[Y] = exp(loc + 0.5 * scale^2)
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "LogNormal",
        loc_transform: Union[str, Callable] = "none",
        scale_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["loc", "scale"], eps=eps)
        self.loc_transform = loc_transform
        self.scale_transform = scale_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError(
                "LogNormalDistribution requires strictly positive targets."
            )

        loc = self.get_transform(self.loc_transform)(predictions[:, 0])
        scale = self.get_transform(self.scale_transform)(predictions[:, 1])

        return -dist.LogNormal(loc=loc, scale=scale).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        loc = params[:, 0]
        scale = params[:, 1]
        return torch.exp(loc + 0.5 * scale.square())

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        loc, scale = params[:, 0], params[:, 1]
        variance = torch.expm1(scale.square()) * torch.exp(2.0 * loc + scale.square())
        return torch.sqrt(torch.clamp(variance, min=0.0))


class WeibullDistribution(ParameterizedObjective):
    """
    Weibull with parameters [scale, shape].

    PyTorch parameterization:
        Weibull(scale=scale, concentration=shape)

    Notes
    -----
    * Targets must be strictly positive.
    * `predict_point()` returns the conditional mean:
          E[Y] = scale * Gamma(1 + 1/shape)
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "Weibull",
        scale_transform: Union[str, Callable] = "positive",
        shape_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["scale", "shape"], eps=eps)
        self.scale_transform = scale_transform
        self.shape_transform = shape_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("WeibullDistribution requires strictly positive targets.")

        scale = self.get_transform(self.scale_transform)(predictions[:, 0])
        shape = self.get_transform(self.shape_transform)(predictions[:, 1])

        return -dist.Weibull(scale=scale, concentration=shape).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        scale = params[:, 0]
        shape = params[:, 1]
        return scale * torch.exp(
            torch.lgamma(1.0 + 1.0 / torch.clamp(shape, min=self.eps))
        )

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        scale, shape = params[:, 0], params[:, 1]
        inverse_shape = 1.0 / torch.clamp(shape, min=self.eps)
        first = torch.exp(torch.lgamma(1.0 + inverse_shape))
        second = torch.exp(torch.lgamma(1.0 + 2.0 * inverse_shape))
        variance = scale.square() * (second - first.square())
        return torch.sqrt(torch.clamp(variance, min=0.0))


class LogLogisticDistribution(ParameterizedObjective):
    """
    Log-Logistic with parameters [scale, shape].

    Density:
        f(y) = (shape / scale) * (y / scale)^(shape - 1) / (1 + (y / scale)^shape)^2
    for y > 0.

    Notes
    -----
    * Targets must be strictly positive.
    * `predict_point()` returns the median, which equals `scale`.
      (The mean exists only for shape > 1.)
    """

    point_summary = "mean"

    def __init__(
        self,
        name: str = "LogLogistic",
        scale_transform: Union[str, Callable] = "positive",
        shape_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["scale", "shape"], eps=eps)
        self.scale_transform = scale_transform
        self.shape_transform = shape_transform

    def compute_loss_per_sample(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError(
                "LogLogisticDistribution requires strictly positive targets."
            )

        scale = self.get_transform(self.scale_transform)(predictions[:, 0])
        shape = self.get_transform(self.shape_transform)(predictions[:, 1])

        log_y = torch.log(torch.clamp(y, min=self.eps))
        log_scale = torch.log(torch.clamp(scale, min=self.eps))
        z = log_y - log_scale
        bz = shape * z

        log_pdf = (
            torch.log(torch.clamp(shape, min=self.eps))
            - log_scale
            + (shape - 1.0) * z
            - 2.0 * F.softplus(bz)
        )

        return -log_pdf

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        # Median of the log-logistic is exactly scale.
        return params[:, 0]

    def predict_spread(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        scale, shape = params[:, 0], params[:, 1]
        angle = torch.pi / torch.clamp(shape, min=self.eps)
        mean = scale * angle / torch.sin(angle)
        second = scale.square() * (2.0 * angle) / torch.sin(2.0 * angle)
        variance = second - mean.square()
        spread = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.where(shape > 2.0, spread, torch.full_like(spread, torch.nan))
