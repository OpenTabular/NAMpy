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
        dim: Optional[int] = None,  # alias
        name: str = "Dirichlet",
        concentration_transform: Union[str, Callable] = "positive",
        target_eps: float = 1e-8,
        eps: float = 1e-8,
    ):
        k = n_dim if n_dim is not None else dim
        if k is None:
            # Keep constructor callable, but fail clearly for training-time usage.
            param_names = ["concentration"]
            self._n_dim = None
        else:
            k = int(k)
            if k < 2:
                raise ValueError("DirichletDistribution requires n_dim >= 2.")
            param_names = [f"alpha_{i}" for i in range(k)]
            self._n_dim = k

        super().__init__(name=name, param_names=param_names, eps=eps)
        self.concentration_transform = concentration_transform
        self.target_eps = float(target_eps)

    @property
    def n_dim(self) -> Optional[int]:
        return self._n_dim

    def _check_dim(self, predictions: torch.Tensor):
        if self._n_dim is None:
            raise ValueError(
                "DirichletDistribution requires `n_dim` (or `dim`) at construction "
                "so `param_count` matches the model output dimension. "
                "Example: DirichletDistribution(n_dim=y.shape[1])."
            )
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
            # Backwards-compatible constructor, but not sufficient for model width.
            self._num_classes = None
            param_names = ["probs"]
        else:
            k = int(num_classes)
            if k < 2:
                raise ValueError("CategoricalDistribution requires num_classes >= 2.")
            self._num_classes = k
            param_names = [f"class_{i}" for i in range(k)]

        super().__init__(name=name, param_names=param_names, eps=eps)
        self.probs_transform = prob_transform

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes

    def _check_dim(self, predictions: torch.Tensor):
        if self._num_classes is None:
            raise ValueError(
                "CategoricalDistribution requires `num_classes` at construction so "
                "`param_count` matches the model output dimension. "
                "Example: CategoricalDistribution(num_classes=K)."
            )
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
    enforce_monotonic : bool, default=False
        If True, `forward()` maps raw predictions to monotone quantiles via:
            q0 = raw0
            increments = softplus(raw[1:])
            q = q0 + cumsum(increments)
        This changes the meaning of the raw parameterization, so default is False
        to preserve backwards compatibility.
    """

    def __init__(
        self,
        name: str = "Quantile",
        quantiles: Optional[Sequence[float]] = None,
        enforce_monotonic: bool = False,
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