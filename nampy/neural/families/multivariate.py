"""Torch distributional families: multivariate."""

from typing import Callable, Optional, Union

import torch
import torch.distributions as dist

from ..objectives.parameterized import ParameterizedObjective


class DirichletDistribution(ParameterizedObjective):
    """
    Dirichlet with K concentration parameters.

    IMPORTANT
    ---------
    This family needs a known output dimension K. Pass e.g.
        DirichletDistribution(n_dim=y.shape[1])
    so `param_count` matches the model output width.
    """

    point_summary = "mean"

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

    def compute_loss_per_sample(self, predictions, y_true):
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
        return -dist.Dirichlet(concentration=concentration).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        """Dirichlet mean = alpha / sum(alpha)."""
        alpha = predictions if transformed else self.forward(predictions)
        alpha = self._ensure_2d_predictions(alpha)
        return alpha / torch.clamp(alpha.sum(dim=1, keepdim=True), min=self.eps)


class MultivariateNormalDiagDistribution(ParameterizedObjective):
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

    point_summary = "mean"

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

    def compute_loss_per_sample(self, predictions, y_true):
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
        return -mvn.log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        params = predictions if transformed else self.forward(predictions)
        params = self._ensure_2d_predictions(params)
        return params[:, : self._n_dim]
