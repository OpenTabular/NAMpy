#utils/distributions.py
import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, np.ndarray]


class BaseDistribution(torch.nn.Module):
    """
    Base class for distributional regression families.

    Subclasses should implement:
      - compute_loss(predictions, y_true)
    and may override:
      - forward(predictions)                -> transformed parameters
      - predict_point(predictions, transformed=False)
      - evaluate_nll(y_true, y_pred)

    Notes
    -----
    * `predictions` in `compute_loss` are assumed to be raw network outputs.
    * `forward(predictions)` returns transformed parameters (e.g. positive scales).
    * `param_count` is the required network output dimension for fixed-dimension families.
      For some families (Dirichlet/Categorical), this may depend on user-provided kwargs.
    """

    target_dtype = torch.float32

    def __init__(self, name: str, param_names: Sequence[str], eps: float = 1e-6):
        super().__init__()
        self._name = name
        self.param_names = list(param_names)
        self.param_count = len(self.param_names)
        self.eps = float(eps)

        self.predefined_transforms = {
            "positive": lambda x: F.softplus(x) + self.eps,
            "strictly_positive": lambda x: F.softplus(x) + self.eps,
            "none": lambda x: x,
            "identity": lambda x: x,
            "square": lambda x: x.square() + self.eps,
            "exp": lambda x: torch.exp(torch.clamp(x, min=-40.0, max=40.0)) + self.eps,
            "sqrt": lambda x: torch.sqrt(torch.clamp(x, min=self.eps)),
            "probabilities": lambda x: torch.softmax(x, dim=-1),
            "log": lambda x: torch.log(torch.clamp(x, min=self.eps)),
            # Monotone transform for quantiles / cutpoints:
            "sort": lambda x: torch.cumsum(F.softplus(x), dim=-1),
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameter_count(self) -> int:
        return self.param_count

    def get_transform(self, transform_name: Union[str, Callable]) -> Callable:
        """Return a transform callable by name or pass through custom callables."""
        if callable(transform_name):
            return transform_name
        if transform_name not in self.predefined_transforms:
            raise ValueError(
                f"Unknown transform {transform_name!r}. "
                f"Available: {sorted(self.predefined_transforms.keys())}"
            )
        return self.predefined_transforms[transform_name]

    # ------------------------------------------------------------------
    # Shape / dtype helpers
    # ------------------------------------------------------------------

    def _ensure_2d_predictions(self, predictions: TensorLike) -> torch.Tensor:
        if not torch.is_tensor(predictions):
            predictions = torch.as_tensor(predictions, dtype=torch.float32)
        else:
            predictions = predictions.float()

        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(-1)
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions must be 1D or 2D; got shape {tuple(predictions.shape)}"
            )
        return predictions

    def _squeeze_target_last_singleton(self, y_true: TensorLike) -> torch.Tensor:
        if not torch.is_tensor(y_true):
            y_true = torch.as_tensor(y_true, dtype=self.target_dtype)
        else:
            y_true = y_true.to(dtype=self.target_dtype)

        # Accept [N, 1] and squeeze to [N]
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        return y_true

    def _validate_batch_match(self, predictions: torch.Tensor, y_true: torch.Tensor):
        if predictions.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Batch size mismatch: predictions has {predictions.shape[0]} rows, "
                f"y_true has {y_true.shape[0]} rows."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_loss(self, predictions: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_loss().")

    def evaluate_nll(self, y_true: TensorLike, y_pred: TensorLike):
        """
        Evaluate negative log-likelihood from transformed or raw predictions.

        Notes
        -----
        This method assumes `y_pred` is in the same format used by `compute_loss`.
        In your codebase, `SklearnBaseLSS.evaluate()` already computes NLL directly
        via `compute_loss` on raw predictions, so this method is mostly a utility.
        """
        y_true_tensor = torch.as_tensor(y_true, dtype=self.target_dtype)
        y_pred_tensor = torch.as_tensor(y_pred, dtype=torch.float32)
        nll_loss_tensor = self.compute_loss(y_pred_tensor, y_true_tensor)
        return {"NLL": float(nll_loss_tensor.detach().cpu().item())}

    def forward(self, predictions: TensorLike) -> torch.Tensor:
        """
        Transform raw network outputs into valid distribution parameters.

        Default implementation applies one transform per parameter column using
        attributes named `<param_name>_transform`.
        """
        predictions = self._ensure_2d_predictions(predictions)

        if predictions.shape[1] != self.param_count:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.param_count} raw parameters, "
                f"got predictions with shape {tuple(predictions.shape)}."
            )

        cols = []
        for idx, param_name in enumerate(self.param_names):
            transform_spec = getattr(self, f"{param_name}_transform", "none")
            transform_fn = self.get_transform(transform_spec)
            cols.append(transform_fn(predictions[:, idx]).unsqueeze(1))
        return torch.cat(cols, dim=1)

    def predict_point(self, predictions: TensorLike, transformed: bool = False) -> torch.Tensor:
        """
        Optional point prediction (default = first transformed parameter).
        Subclasses can override for family-specific meanings.
        """
        params = predictions if transformed else self.forward(predictions)
        if not torch.is_tensor(params):
            params = torch.as_tensor(params, dtype=torch.float32)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]


class NormalDistribution(BaseDistribution):
    """
    Gaussian with parameters [mean, scale].

    Notes
    -----
    `scale` is the standard deviation (NOT the variance).
    """

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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        scale = self.get_transform(self.scale_transform)(predictions[:, 1])
        return -dist.Normal(loc=mean, scale=scale).log_prob(y).mean()

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


class PoissonDistribution(BaseDistribution):
    """Poisson with parameter [rate]."""

    def __init__(
        self,
        name: str = "Poisson",
        rate_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["rate"], eps=eps)
        self.rate_transform = rate_transform

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("PoissonDistribution requires non-negative targets.")

        rate = self.get_transform(self.rate_transform)(predictions[:, 0])
        return -dist.Poisson(rate=rate).log_prob(y).mean()

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


class InverseGammaDistribution(BaseDistribution):
    """Inverse-Gamma with parameters [shape, rate]."""

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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("InverseGammaDistribution requires strictly positive targets.")

        shape = self.get_transform(self.shape_transform)(predictions[:, 0])
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])
        return -dist.InverseGamma(concentration=shape, rate=rate).log_prob(y).mean()

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


class BetaDistribution(BaseDistribution):
    """Beta with parameters [alpha, beta]. Targets must lie in (0, 1)."""

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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        y = torch.clamp(y, min=self.target_eps, max=1.0 - self.target_eps)
        alpha = self.get_transform(self.alpha_transform)(predictions[:, 0])
        beta = self.get_transform(self.beta_transform)(predictions[:, 1])
        return -dist.Beta(concentration1=alpha, concentration0=beta).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        """Beta mean = alpha / (alpha + beta)."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        alpha = params[:, 0]
        beta = params[:, 1]
        return alpha / torch.clamp(alpha + beta, min=self.eps)


class DirichletDistribution(BaseDistribution):
    """
    Dirichlet with K concentration parameters.

    IMPORTANT
    ---------
    This family needs a known output dimension K. Pass e.g.
        DirichletDistribution(n_dim=y.shape[1])
    so `param_count` matches the model output width.
    """

    def __init__(
        self,
        n_dim: Optional[int] = None,
        name: str = "Dirichlet",
        concentration_transform: Union[str, Callable] = "positive",
        target_eps: float = 1e-8,
        eps: float = 1e-8,
    ):
        k = n_dim
        if k is None:
            raise ValueError(
                "DirichletDistribution requires `n_dim` at construction "
                "so `param_count` matches the model output dimension. "
                "Example: DirichletDistribution(n_dim=y.shape[1])."
            )
        k = int(k)
        if k < 2:
            raise ValueError("DirichletDistribution requires n_dim >= 2.")
        param_names = [f"alpha_{i}" for i in range(k)]
        self._n_dim = k

        super().__init__(name=name, param_names=param_names, eps=eps)
        self.concentration_transform = concentration_transform
        self.target_eps = float(target_eps)

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def _check_dim(self, predictions: torch.Tensor):
        if predictions.shape[1] != self._n_dim:
            raise ValueError(
                f"DirichletDistribution expected {self._n_dim} parameters, got "
                f"predictions with shape {tuple(predictions.shape)}."
            )

    def forward(self, predictions):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)
        tfm = self.get_transform(self.concentration_transform)
        return tfm(predictions)

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)

        if not torch.is_tensor(y_true):
            y = torch.as_tensor(y_true, dtype=torch.float32)
        else:
            y = y_true.float()

        # Accept [N, K], [N, 1, K]
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        if y.ndim != 2:
            raise ValueError(
                f"Dirichlet targets must have shape [N, K] (or [N,1,K]); got {tuple(y.shape)}"
            )
        self._validate_batch_match(predictions, y)
        if y.shape[1] != predictions.shape[1]:
            raise ValueError(
                f"Dirichlet target dimension {y.shape[1]} != prediction dimension {predictions.shape[1]}"
            )

        # Clamp and renormalize to simplex
        y = torch.clamp(y, min=self.target_eps)
        y = y / torch.clamp(y.sum(dim=1, keepdim=True), min=self.target_eps)

        concentration = self.forward(predictions)
        return -dist.Dirichlet(concentration=concentration).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        """Dirichlet mean = alpha / sum(alpha)."""
        alpha = predictions if transformed else self.forward(predictions)
        alpha = self._ensure_2d_predictions(alpha)
        return alpha / torch.clamp(alpha.sum(dim=1, keepdim=True), min=self.eps)


class GammaDistribution(BaseDistribution):
    """Gamma with parameters [shape, rate]. Targets must be > 0."""

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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("GammaDistribution requires strictly positive targets.")

        shape = self.get_transform(self.shape_transform)(predictions[:, 0])
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])
        return -dist.Gamma(concentration=shape, rate=rate).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        """Gamma mean = shape / rate."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0] / torch.clamp(params[:, 1], min=self.eps)


class StudentTDistribution(BaseDistribution):
    """Student-t with parameters [df, loc, scale]."""

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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        df = self.get_transform(self.df_transform)(predictions[:, 0]) + self.min_df
        loc = self.get_transform(self.loc_transform)(predictions[:, 1])
        scale = self.get_transform(self.scale_transform)(predictions[:, 2])
        return -dist.StudentT(df=df, loc=loc, scale=scale).log_prob(y).mean()

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


class NegativeBinomialDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("NegativeBinomialDistribution requires non-negative targets.")

        mean = self.get_transform(self.mean_transform)(predictions[:, 0])
        alpha = self.get_transform(self.dispersion_transform)(predictions[:, 1])

        total_count = 1.0 / torch.clamp(alpha, min=self.eps)  # r
        probs = mean / torch.clamp(mean + total_count, min=self.eps)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        nb = dist.NegativeBinomial(total_count=total_count, probs=probs)
        return -nb.log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        """NB mean parameter."""
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]


class CategoricalDistribution(BaseDistribution):
    """
    Categorical over K classes (returns probabilities in `forward`).

    IMPORTANT
    ---------
    For correct model output width, pass:
        CategoricalDistribution(num_classes=K)
    so `param_count == K`.
    """

    target_dtype = torch.long

    def __init__(
        self,
        num_classes: Optional[int] = None,
        name: str = "Categorical",
        prob_transform: Union[str, Callable] = "probabilities",
        eps: float = 1e-8,
    ):
        if num_classes is None:
            raise ValueError(
                "CategoricalDistribution requires `num_classes` at construction so "
                "`param_count` matches the model output dimension. "
                "Example: CategoricalDistribution(num_classes=K)."
            )
        k = int(num_classes)
        if k < 2:
            raise ValueError("CategoricalDistribution requires num_classes >= 2.")
        self._num_classes = k
        param_names = [f"class_{i}" for i in range(k)]

        super().__init__(name=name, param_names=param_names, eps=eps)
        self.probs_transform = prob_transform

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _check_dim(self, predictions: torch.Tensor):
        if predictions.shape[1] != self._num_classes:
            raise ValueError(
                f"CategoricalDistribution expected {self._num_classes} logits, got "
                f"predictions with shape {tuple(predictions.shape)}."
            )

    def forward(self, predictions):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)
        tfm = self.get_transform(self.probs_transform)
        probs = tfm(predictions)
        return torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)

        if not torch.is_tensor(y_true):
            y = torch.as_tensor(y_true)
        else:
            y = y_true

        # Accept labels [N], [N,1], one-hot/probabilities [N,K], [N,1,K]
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        elif y.ndim == 2 and y.shape[1] == predictions.shape[1]:
            y = torch.argmax(y, dim=1)
        elif y.ndim != 1:
            raise ValueError(
                f"Categorical targets must be [N], [N,1], or [N,K]; got {tuple(y.shape)}"
            )

        y = y.long()
        self._validate_batch_match(predictions, y)
        probs = self.forward(predictions)
        return -dist.Categorical(probs=probs).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        probs = predictions if transformed else self.forward(predictions)
        probs = self._ensure_2d_predictions(probs)
        return torch.argmax(probs, dim=1)


class Quantile(BaseDistribution):
    """
    Quantile regression (pinball loss).

    Parameters
    ----------
    quantiles : sequence of float
        Quantiles in (0,1). Example: [0.1, 0.5, 0.9]
    enforce_monotonic : bool, default=True
        If True, `forward()` maps raw predictions to monotone quantiles via:
            q0 = raw0
            increments = softplus(raw[1:])
            q = q0 + cumsum(increments)
        This changes the meaning of the raw parameterization but avoids
        quantile crossing, which is typically desired in practice.
    """

    def __init__(
        self,
        name: str = "Quantile",
        quantiles: Optional[Sequence[float]] = None,
        enforce_monotonic: bool = True,
        eps: float = 1e-8,
    ):
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]

        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("quantiles must be a non-empty 1D sequence.")
        if np.any(q <= 0.0) or np.any(q >= 1.0):
            raise ValueError("All quantiles must lie strictly between 0 and 1.")
        if len(np.unique(q)) != len(q):
            raise ValueError("quantiles must be unique.")
        if np.any(np.diff(q) < 0):
            raise ValueError("quantiles must be sorted ascending.")

        self.quantiles = [float(v) for v in q]
        self.enforce_monotonic = bool(enforce_monotonic)
        param_names = [f"q_{v:g}" for v in self.quantiles]
        super().__init__(name=name, param_names=param_names, eps=eps)

    def forward(self, predictions):
        predictions = self._ensure_2d_predictions(predictions)
        if predictions.shape[1] != self.param_count:
            raise ValueError(
                f"Quantile expects {self.param_count} outputs, got {tuple(predictions.shape)}"
            )

        if not self.enforce_monotonic:
            return predictions

        q0 = predictions[:, :1]
        if predictions.shape[1] == 1:
            return q0
        inc = F.softplus(predictions[:, 1:])
        q_rest = q0 + torch.cumsum(inc, dim=1)
        return torch.cat([q0, q_rest], dim=1)

    def compute_loss(self, predictions, y_true):
        preds = self.forward(predictions)  # supports optional monotonic transform
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(preds, y)

        # Broadcast y to [N, Q]
        y = y.unsqueeze(1)
        q = torch.tensor(self.quantiles, dtype=preds.dtype, device=preds.device).view(1, -1)
        errors = y - preds
        loss = torch.maximum((q - 1.0) * errors, q * errors)
        return loss.sum(dim=1).mean()

    def predict_point(self, predictions, transformed: bool = False):
        preds = predictions if transformed else self.forward(predictions)
        preds = self._ensure_2d_predictions(preds)
        if 0.5 in self.quantiles:
            idx = self.quantiles.index(0.5)
            return preds[:, idx]
        return preds.mean(dim=1)


class RobustNormalDistribution(BaseDistribution):
    """
    Robustified Gaussian likelihood with parameters [mean, scale].

    The robustness transformation is:
        loglik_rob = log((1 + exp(loglik + rob)) / (1 + exp(rob)))
    This downweights very unlikely observations while remaining smooth.
    """

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

    def compute_loss(self, predictions, y_true):
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
            rob_t = torch.tensor(self.rob, device=log_likelihood.device, dtype=log_likelihood.dtype)
            log_num = torch.logaddexp(torch.zeros_like(log_likelihood), log_likelihood + rob_t)
            log_den = torch.logaddexp(torch.tensor(0.0, device=rob_t.device, dtype=rob_t.dtype), rob_t)
            log_likelihood = log_num - log_den

        return -log_likelihood.mean()

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

# ----------------------------------------------------------------------
# Additional drop-in families for nampy/utils/distributions.py
# ----------------------------------------------------------------------

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


class LogNormalDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("LogNormalDistribution requires strictly positive targets.")

        loc = self.get_transform(self.loc_transform)(predictions[:, 0])
        scale = self.get_transform(self.scale_transform)(predictions[:, 1])

        return -dist.LogNormal(loc=loc, scale=scale).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        loc = params[:, 0]
        scale = params[:, 1]
        return torch.exp(loc + 0.5 * scale.square())


class WeibullDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("WeibullDistribution requires strictly positive targets.")

        scale = self.get_transform(self.scale_transform)(predictions[:, 0])
        shape = self.get_transform(self.shape_transform)(predictions[:, 1])

        return -dist.Weibull(scale=scale, concentration=shape).log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        scale = params[:, 0]
        shape = params[:, 1]
        return scale * torch.exp(torch.lgamma(1.0 + 1.0 / torch.clamp(shape, min=self.eps)))


class LogLogisticDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y <= 0):
            raise ValueError("LogLogisticDistribution requires strictly positive targets.")

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

        return -log_pdf.mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        # Median of the log-logistic is exactly scale.
        return params[:, 0]


class ZeroInflatedPoissonDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(predictions, y)

        if torch.any(y < 0):
            raise ValueError("ZeroInflatedPoissonDistribution requires non-negative targets.")

        zero_prob = self.get_transform(self.zero_prob_transform)(predictions[:, 0])
        zero_prob = torch.clamp(zero_prob, min=self.eps, max=1.0 - self.eps)
        rate = self.get_transform(self.rate_transform)(predictions[:, 1])

        pois = dist.Poisson(rate=rate)
        log_pois = pois.log_prob(y)
        log_pois0 = pois.log_prob(torch.zeros_like(y))

        log_prob_zero = torch.logaddexp(torch.log(zero_prob), torch.log1p(-zero_prob) + log_pois0)
        log_prob_pos = torch.log1p(-zero_prob) + log_pois

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob.mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        rate = params[:, 1]
        return (1.0 - zero_prob) * rate


class ZeroInflatedNegativeBinomialDistribution(BaseDistribution):
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

    def __init__(
        self,
        name: str = "ZeroInflatedNegativeBinomial",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["zero_prob", "mean", "dispersion"], eps=eps)
        self.zero_prob_transform = zero_prob_transform
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform

    def compute_loss(self, predictions, y_true):
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

        log_prob_zero = torch.logaddexp(torch.log(zero_prob), torch.log1p(-zero_prob) + log_nb0)
        log_prob_pos = torch.log1p(-zero_prob) + log_nb

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob.mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        mean = params[:, 1]
        return (1.0 - zero_prob) * mean


class HurdlePoissonDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
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

        log_trunc_norm = torch.log(torch.clamp(1.0 - torch.exp(log_pois0), min=self.eps))
        log_prob_zero = torch.log(zero_prob)
        log_prob_pos = torch.log1p(-zero_prob) + log_pois - log_trunc_norm

        log_prob = torch.where(y == 0, log_prob_zero, log_prob_pos)
        return -log_prob.mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        zero_prob = params[:, 0]
        rate = params[:, 1]
        trunc_mean = rate / torch.clamp(1.0 - torch.exp(-rate), min=self.eps)
        return (1.0 - zero_prob) * trunc_mean


class HurdleNegativeBinomialDistribution(BaseDistribution):
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

    def __init__(
        self,
        name: str = "HurdleNegativeBinomial",
        zero_prob_transform: Union[str, Callable] = _unit_interval_transform,
        mean_transform: Union[str, Callable] = "positive",
        dispersion_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        super().__init__(name=name, param_names=["zero_prob", "mean", "dispersion"], eps=eps)
        self.zero_prob_transform = zero_prob_transform
        self.mean_transform = mean_transform
        self.dispersion_transform = dispersion_transform

    def compute_loss(self, predictions, y_true):
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
        return -log_prob.mean()

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

# ----------------------------------------------------------------------
# Additional advanced families for nampy/utils/distributions.py
# ----------------------------------------------------------------------

def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus for x > 0."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x + torch.log(-torch.expm1(-x))


class TweedieDistribution(BaseDistribution):
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

    def compute_loss(self, predictions, y_true):
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

        return -log_prob.mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, 0]


class OrdinalCumulativeLogitDistribution(BaseDistribution):
    """
    Proportional-odds ordinal regression with global ordered cutpoints.

    Model
    -----
    The network predicts a single latent location per sample:
        eta(x) in R

    The family holds global ordered cutpoints:
        c_1 < c_2 < ... < c_{K-1}

    Then:
        P(Y <= k | x) = sigmoid(c_k - eta(x))

    This is often the cleanest ordinal head for tabular models because the network
    only needs to output one score per sample, while the thresholds remain global.

    Parameters
    ----------
    num_classes : int
        Number of ordered classes K >= 2.

    Notes
    -----
    * `param_count == 1`, so the backbone only needs one output dimension.
    * `forward()` returns class probabilities of shape [N, K].
    * `predict_point()` returns the most likely class index.
    """

    target_dtype = torch.long

    def __init__(
        self,
        num_classes: Optional[int] = None,
        name: str = "OrdinalCumulativeLogit",
        eps: float = 1e-8,
    ):
        if num_classes is None:
            raise ValueError(
                "OrdinalCumulativeLogitDistribution requires `num_classes` at construction."
            )
        k = int(num_classes)
        if k < 2:
            raise ValueError("OrdinalCumulativeLogitDistribution requires num_classes >= 2.")

        super().__init__(name=name, param_names=["eta"], eps=eps)
        self._num_classes = k

        # Initialize ordered cutpoints roughly centered around 0.
        init_cuts = torch.linspace(-(k - 2) / 2.0, (k - 2) / 2.0, k - 1)
        self.first_cutpoint = torch.nn.Parameter(init_cuts[:1].clone())

        if k > 2:
            init_diffs = init_cuts[1:] - init_cuts[:-1]
            self.raw_cutpoint_increments = torch.nn.Parameter(
                _inverse_softplus(init_diffs).clone()
            )
        else:
            self.raw_cutpoint_increments = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _ordered_cutpoints(self) -> torch.Tensor:
        if self.raw_cutpoint_increments is None:
            return self.first_cutpoint
        inc = F.softplus(self.raw_cutpoint_increments)
        rest = self.first_cutpoint + torch.cumsum(inc, dim=0)
        return torch.cat([self.first_cutpoint, rest], dim=0)

    def get_cutpoints(self) -> torch.Tensor:
        """Return current ordered cutpoints as a detached tensor."""
        return self._ordered_cutpoints().detach()

    def forward(self, predictions):
        predictions = self._ensure_2d_predictions(predictions)
        if predictions.shape[1] != 1:
            raise ValueError(
                f"OrdinalCumulativeLogitDistribution expects raw predictions with shape (N, 1), "
                f"got {tuple(predictions.shape)}."
            )

        eta = predictions[:, 0:1]  # [N,1]
        cutpoints = self._ordered_cutpoints().to(device=eta.device, dtype=eta.dtype)  # [K-1]

        cum_probs = torch.sigmoid(cutpoints.unsqueeze(0) - eta)  # [N, K-1]

        probs = []
        probs.append(cum_probs[:, 0:1])  # P(Y=0)
        for j in range(1, self.num_classes - 1):
            probs.append(cum_probs[:, j:j+1] - cum_probs[:, j-1:j])
        probs.append(1.0 - cum_probs[:, -1:])  # P(Y=K-1)

        probs = torch.cat(probs, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=self.eps)
        return probs

    def compute_loss(self, predictions, y_true):
        probs = self.forward(predictions)

        if not torch.is_tensor(y_true):
            y = torch.as_tensor(y_true)
        else:
            y = y_true

        # Accept [N], [N,1], [N,K], [N,1,K]
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        elif y.ndim == 2 and y.shape[1] == self.num_classes:
            y = torch.argmax(y, dim=1)
        elif y.ndim != 1:
            raise ValueError(
                f"Ordinal targets must be [N], [N,1], or [N,K]; got {tuple(y.shape)}"
            )

        y = y.long().to(device=probs.device)
        self._validate_batch_match(probs, y)

        if torch.any((y < 0) | (y >= self.num_classes)):
            raise ValueError(
                f"Ordinal targets must lie in [0, {self.num_classes - 1}]."
            )

        gathered = probs.gather(1, y.unsqueeze(1)).squeeze(1)
        return -torch.log(torch.clamp(gathered, min=self.eps)).mean()

    def predict_point(self, predictions, transformed: bool = False):
        probs = predictions if transformed else self.forward(predictions)
        probs = self._ensure_2d_predictions(probs)
        return torch.argmax(probs, dim=1)


class MultivariateNormalDiagDistribution(BaseDistribution):
    """
    Multivariate Normal with diagonal covariance for K-dimensional targets.

    Raw network output layout
    -------------------------
    [loc_0, ..., loc_{K-1}, scale_0, ..., scale_{K-1}]

    Notes
    -----
    * Requires `n_dim` (or `dim`) at construction.
    * Assumes conditional independence across target dimensions given x.
    * `predict_point()` returns the mean vector of shape [N, K].
    """

    def __init__(
        self,
        n_dim: Optional[int] = None,
        dim: Optional[int] = None,
        name: str = "MultivariateNormalDiag",
        loc_transform: Union[str, Callable] = "none",
        scale_transform: Union[str, Callable] = "positive",
        eps: float = 1e-8,
    ):
        k = n_dim if n_dim is not None else dim
        if k is None:
            raise ValueError(
                "MultivariateNormalDiagDistribution requires `n_dim` (or `dim`) "
                "at construction."
            )
        k = int(k)
        if k < 2:
            raise ValueError("MultivariateNormalDiagDistribution requires n_dim >= 2.")

        param_names = [f"loc_{i}" for i in range(k)] + [f"scale_{i}" for i in range(k)]
        super().__init__(name=name, param_names=param_names, eps=eps)

        self._n_dim = k
        self.loc_transform = loc_transform
        self.scale_transform = scale_transform

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def _check_dim(self, predictions: torch.Tensor):
        if predictions.shape[1] != 2 * self._n_dim:
            raise ValueError(
                f"MultivariateNormalDiagDistribution expected {2 * self._n_dim} raw parameters, "
                f"got {tuple(predictions.shape)}."
            )

    def forward(self, predictions):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)

        loc_raw = predictions[:, : self._n_dim]
        scale_raw = predictions[:, self._n_dim :]

        loc = self.get_transform(self.loc_transform)(loc_raw)
        scale = self.get_transform(self.scale_transform)(scale_raw)

        return torch.cat([loc, scale], dim=1)

    def compute_loss(self, predictions, y_true):
        predictions = self._ensure_2d_predictions(predictions)
        self._check_dim(predictions)

        if not torch.is_tensor(y_true):
            y = torch.as_tensor(y_true, dtype=torch.float32)
        else:
            y = y_true.float()

        # Accept [N, K], [N,1,K]
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        if y.ndim != 2:
            raise ValueError(
                f"MultivariateNormalDiag targets must have shape [N, K] (or [N,1,K]); got {tuple(y.shape)}"
            )
        self._validate_batch_match(predictions, y)
        if y.shape[1] != self._n_dim:
            raise ValueError(
                f"MultivariateNormalDiag target dimension {y.shape[1]} != expected {self._n_dim}."
            )

        loc = self.get_transform(self.loc_transform)(predictions[:, : self._n_dim])
        scale = self.get_transform(self.scale_transform)(predictions[:, self._n_dim :])

        mvn = dist.Independent(dist.Normal(loc=loc, scale=scale), 1)
        return -mvn.log_prob(y).mean()

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, : self._n_dim]