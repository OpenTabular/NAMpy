"""Torch distributional families: quantile."""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .parameterized import ParameterizedObjective


class Quantile(ParameterizedObjective):
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

    def compute_loss_per_sample(self, predictions, y_true):
        preds = self.forward(predictions)  # supports optional monotonic transform
        y = self._squeeze_target_last_singleton(y_true).float()
        self._validate_batch_match(preds, y)

        # Broadcast y to [N, Q]
        y = y.unsqueeze(1)
        q = torch.tensor(self.quantiles, dtype=preds.dtype, device=preds.device).view(
            1, -1
        )
        errors = y - preds
        loss = torch.maximum((q - 1.0) * errors, q * errors)
        return loss.sum(dim=1)

    @property
    def point_summary(self):
        return "median" if 0.5 in self.quantiles else "mean_of_quantiles"

    def predict_point(self, predictions, transformed: bool = False):
        preds = predictions if transformed else self.forward(predictions)
        preds = self._ensure_2d_predictions(preds)
        if 0.5 in self.quantiles:
            idx = self.quantiles.index(0.5)
            return preds[:, idx]
        return preds.mean(dim=1)
