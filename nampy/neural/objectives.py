"""Training objectives independent of neural forward architectures."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class NeuralObjective(nn.Module):
    """Own output semantics, target preparation, loss, and task metrics."""

    kind = "objective"
    datamodule_regression = True
    allows_offset = True

    def __init__(self, output_dim: int):
        super().__init__()
        if int(output_dim) < 1:
            raise ValueError("Objective output_dim must be positive.")
        self.output_dim = int(output_dim)

    def prepare_targets(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_values(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def reduce_loss(loss_values: torch.Tensor, sample_weight=None) -> torch.Tensor:
        if loss_values.ndim == 0:
            if sample_weight is not None:
                raise ValueError(
                    "Weighted training requires a loss with one value per sample."
                )
            return loss_values
        per_sample = loss_values.reshape(loss_values.shape[0], -1).mean(dim=1)
        if sample_weight is None:
            return per_sample.mean()
        weights = sample_weight.reshape(-1).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        total_weight = torch.sum(weights)
        if total_weight <= 0:
            return per_sample.sum() * 0.0
        return torch.sum(per_sample * weights) / total_weight

    def compute_loss(self, predictions, targets, sample_weight=None):
        prepared = self.prepare_targets(predictions, targets)
        values = self.loss_values(predictions, prepared)
        return self.reduce_loss(values, sample_weight)

    def transform(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions

    def log_metrics(
        self,
        owner,
        prefix: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        data_loss: torch.Tensor,
    ) -> None:
        del owner, prefix, predictions, targets, data_loss


class RegressionObjective(NeuralObjective):
    kind = "regression"

    def __init__(self, output_dim: int = 1, loss_fct: Any = None):
        super().__init__(output_dim)
        self.loss_fct = loss_fct
        self.uses_default_loss = loss_fct is None

    def prepare_targets(self, predictions, targets):
        y = targets.to(dtype=predictions.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.shape != predictions.shape:
            raise ValueError(
                f"Regression predictions have shape {tuple(predictions.shape)}, but "
                f"targets have shape {tuple(y.shape)}."
            )
        return y

    def loss_values(self, predictions, targets):
        if self.uses_default_loss:
            return F.mse_loss(predictions, targets, reduction="none")
        return self.loss_fct(predictions, targets)

    def log_metrics(self, owner, prefix, predictions, targets, data_loss):
        del predictions, targets
        if prefix == "test":
            owner.log(
                "test_rmse",
                torch.sqrt(data_loss),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )


class BinaryObjective(NeuralObjective):
    kind = "binary"
    datamodule_regression = False

    def __init__(self):
        super().__init__(1)
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
        self.precision = torchmetrics.Precision(task="binary")

    def prepare_targets(self, predictions, targets):
        y = targets.to(dtype=predictions.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.shape != predictions.shape:
            y = y.view_as(predictions)
        return y

    def loss_values(self, predictions, targets):
        return F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

    def transform(self, predictions):
        probability = torch.sigmoid(predictions)
        return torch.cat([1.0 - probability, probability], dim=1)

    def log_metrics(self, owner, prefix, predictions, targets, data_loss):
        del data_loss
        probabilities = torch.sigmoid(predictions).view(-1)
        y = targets.view(-1).long()
        for name, metric, prog_bar in (
            ("acc", self.acc, True),
            ("auroc", self.auroc, False),
            ("precision", self.precision, False),
        ):
            owner.log(
                f"{prefix}_{name}",
                metric(probabilities, y),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
                logger=True,
            )


class MulticlassObjective(NeuralObjective):
    kind = "multiclass"
    datamodule_regression = False

    def __init__(self, num_classes: int):
        if int(num_classes) < 3:
            raise ValueError("MulticlassObjective requires at least three classes.")
        super().__init__(num_classes)
        self.num_classes = int(num_classes)
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )

    def prepare_targets(self, predictions, targets):
        del predictions
        return targets.long().view(-1)

    def loss_values(self, predictions, targets):
        return F.cross_entropy(predictions, targets, reduction="none")

    def transform(self, predictions):
        return torch.softmax(predictions, dim=1)

    def log_metrics(self, owner, prefix, predictions, targets, data_loss):
        del data_loss
        y = targets.view(-1).long()
        probabilities = torch.softmax(predictions, dim=1)
        for name, metric, values, prog_bar in (
            ("acc", self.acc, predictions, True),
            ("auroc", self.auroc, probabilities, False),
            ("precision", self.precision, predictions, False),
        ):
            owner.log(
                f"{prefix}_{name}",
                metric(values, y),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
                logger=True,
            )


class DistributionObjective(NeuralObjective):
    kind = "distributional"
    allows_offset = False

    def __init__(self, family):
        super().__init__(family.param_count)
        self.family = family

    def prepare_targets(self, predictions, targets):
        del predictions
        y = targets
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        target_dtype = getattr(self.family, "target_dtype", None)
        if target_dtype is not None:
            y = y.to(dtype=target_dtype)
        return y

    def loss_values(self, predictions, targets):
        return self.family.compute_loss(predictions, targets, reduction="none")

    def transform(self, predictions):
        return self.family(predictions)


def classification_objective(num_classes: int) -> NeuralObjective:
    if int(num_classes) < 2:
        raise ValueError("Classification requires at least two classes.")
    if int(num_classes) == 2:
        return BinaryObjective()
    return MulticlassObjective(num_classes)


def objective_from_legacy(
    *,
    num_classes: int,
    task: str | None,
    lss: bool,
    family,
    loss_fct=None,
) -> NeuralObjective:
    """Translate the former TaskModule arguments at the compatibility boundary."""
    if lss:
        if family is None:
            raise ValueError("Distributional objectives require a family.")
        return DistributionObjective(family)
    requested = None if task is None else str(task).lower()
    if requested is None:
        requested = "regression" if int(num_classes) == 1 else "classification"
    if requested == "regression":
        return RegressionObjective(num_classes, loss_fct=loss_fct)
    if requested == "classification":
        if loss_fct is not None:
            raise ValueError("Custom losses are not supported for classification.")
        return classification_objective(num_classes)
    raise ValueError(
        f"Unsupported supervised task {task!r}; expected 'regression' or "
        "'classification'."
    )


__all__ = [
    "BinaryObjective",
    "DistributionObjective",
    "MulticlassObjective",
    "NeuralObjective",
    "RegressionObjective",
    "classification_objective",
    "objective_from_legacy",
]
