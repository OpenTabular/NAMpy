"""Parameterized neural response objectives: transforms, losses and summaries."""

from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, np.ndarray]


class ParameterizedObjective(torch.nn.Module):
    """
    Base class for parameterized response losses, including likelihoods and quantiles.

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

    @property
    def point_summary(self) -> str:
        return "family_point"
    target_dtype = torch.float32

    @property
    def natural_parameter_names(self):
        """Names of columns returned by the response-parameter transform."""
        return tuple(self.param_names)

    def __init__(self, name: str, param_names: Sequence[str], eps: float = 1e-6):
        super().__init__()
        self._name = name
        self.param_names = list(param_names)
        self.param_count = len(self.param_names)
        self.eps = float(eps)

        self.predefined_transforms = {
            "positive": self._positive_transform,
            "strictly_positive": self._positive_transform,
            "none": self._identity_transform,
            "identity": self._identity_transform,
            "square": self._square_transform,
            "exp": self._exp_transform,
            "sqrt": self._sqrt_transform,
            "probabilities": self._probability_transform,
            "log": self._log_transform,
            # Monotone transform for quantiles / cutpoints:
            "sort": self._sorted_transform,
        }

    def _positive_transform(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + self.eps

    @staticmethod
    def _identity_transform(x: torch.Tensor) -> torch.Tensor:
        return x

    def _square_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.square() + self.eps

    def _exp_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, min=-40.0, max=40.0)) + self.eps

    def _sqrt_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x, min=self.eps))

    @staticmethod
    def _probability_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))

    @staticmethod
    def _sorted_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(F.softplus(x), dim=-1)

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
        elif not predictions.is_floating_point():
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

    def compute_loss(
        self,
        predictions: torch.Tensor,
        y_true: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return a response loss with a shared, weight-compatible reduction."""
        values = self.compute_loss_per_sample(predictions, y_true)
        if values.ndim == 0:
            raise ValueError(
                f"{type(self).__name__}.compute_loss_per_sample() returned a scalar."
            )
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        if reduction == "sum":
            return values.sum()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    def compute_loss_per_sample(
        self, predictions: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement compute_loss_per_sample()."
        )

    def evaluate_nll(self, y_true: TensorLike, y_pred: TensorLike):
        """
        Evaluate negative log-likelihood from transformed or raw predictions.

        Notes
        -----
        This method assumes `y_pred` is in the same format used by `compute_loss`.
        In your codebase, `NeuralLSS.evaluate()` already computes NLL directly
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

    def predict_point(
        self, predictions: TensorLike, transformed: bool = False
    ) -> torch.Tensor:
        """Return a family-defined point functional, if implemented."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define a point summary."
        )

    def predict_spread(
        self, predictions: TensorLike, transformed: bool = False
    ) -> torch.Tensor:
        """Return the conditional standard deviation on the response scale.

        Distributional effect plots use this family-level hook instead of
        interpreting parameter names in the plotting code.  Scalar families
        with a defined standard deviation should override it; families whose
        output is multivariate, categorical, or quantile-only deliberately do
        not invent a scalar spread.
        """
        del predictions, transformed
        raise NotImplementedError(
            f"{type(self).__name__} does not define a scalar conditional "
            "standard deviation."
        )
