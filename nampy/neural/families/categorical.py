"""Torch distributional families: categorical.

References
----------
- McCullagh, P. (1980). Regression Models for Ordinal Data.
  https://doi.org/10.1111/j.2517-6161.1980.tb01109.x

This reference concerns the ordinal cumulative-link branch; categorical softmax is a
separate response map.
"""

from typing import Callable, Optional, Union

import torch
import torch.distributions as dist
import torch.nn.functional as F

from ..objectives.parameterized import ParameterizedObjective


class CategoricalDistribution(ParameterizedObjective):
    """
    Categorical over K classes (returns probabilities in `forward`).

    IMPORTANT
    ---------
    For correct model output width, pass:
        CategoricalDistribution(num_classes=K)
    so `param_count == K`.
    """

    point_summary = "mode"

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

    def compute_loss_per_sample(self, predictions, y_true):
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
        return -dist.Categorical(probs=probs).log_prob(y)

    def predict_point(self, predictions, transformed: bool = False):
        probs = predictions if transformed else self.forward(predictions)
        probs = self._ensure_2d_predictions(probs)
        return torch.argmax(probs, dim=1)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus for x > 0."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x + torch.log(-torch.expm1(-x))


class OrdinalCumulativeLogitDistribution(ParameterizedObjective):
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

    point_summary = "mode"

    @property
    def natural_parameter_names(self):
        return tuple(f"class_{i}" for i in range(self.num_classes))

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
            raise ValueError(
                "OrdinalCumulativeLogitDistribution requires num_classes >= 2."
            )

        super().__init__(name=name, param_names=["eta"], eps=eps)
        self._num_classes = k

        # Initialize ordered cutpoints roughly centered around 0.
        init_cuts = torch.linspace(-(k - 2) / 2.0, (k - 2) / 2.0, k - 1)
        self.first_cutpoint = torch.nn.Parameter(init_cuts[:1].clone())

        self.raw_cutpoint_increments: torch.nn.Parameter | None
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
        cutpoints = self._ordered_cutpoints().to(
            device=eta.device, dtype=eta.dtype
        )  # [K-1]

        cum_probs = torch.sigmoid(cutpoints.unsqueeze(0) - eta)  # [N, K-1]

        probs = []
        probs.append(cum_probs[:, 0:1])  # P(Y=0)
        for j in range(1, self.num_classes - 1):
            probs.append(cum_probs[:, j : j + 1] - cum_probs[:, j - 1 : j])
        probs.append(1.0 - cum_probs[:, -1:])  # P(Y=K-1)

        probs = torch.cat(probs, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=self.eps)
        return probs

    def compute_loss_per_sample(self, predictions, y_true):
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
        return -torch.log(torch.clamp(gathered, min=self.eps))

    def predict_point(self, predictions, transformed: bool = False):
        probs = predictions if transformed else self.forward(predictions)
        probs = self._ensure_2d_predictions(probs)
        return torch.argmax(probs, dim=1)
