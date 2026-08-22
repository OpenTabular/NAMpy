import math
from typing import Any, Type

import lightning as pl
import torch
import torch.nn as nn

from .contracts import harvest_penalties
from .objectives import objective_from_legacy


class TaskModule(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        num_classes=1,  # semantic class count for classification, output dim for regression/LSS
        task: str | None = None,
        lss=False,
        family=None,
        objective=None,
        loss_fct: Any = None,
        optimizer="adam",
        optimizer_kwargs=None,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_decay_factor=0.2,
        lr_schedule="plateau",
        scheduler_monitor="val_loss",
        scheduler_mode="min",
        pretraining=False,
        pretraining_ratio=0.15,
        pretraining_noise=0.1,
        pretraining_feature_mask=True,
        **kwargs,
    ):
        super().__init__()

        self.pretraining = bool(pretraining)
        self.pretraining_ratio = float(pretraining_ratio)
        self.pretraining_noise = float(pretraining_noise)
        self.pretraining_feature_mask = bool(pretraining_feature_mask)
        if self.pretraining:
            self.objective = None
            self.task_kind = "pretraining"
            self.output_dim = int(num_classes)
        else:
            if objective is None:
                objective = objective_from_legacy(
                    num_classes=num_classes,
                    task=task,
                    lss=lss,
                    family=family,
                    loss_fct=loss_fct,
                )
            self.objective = objective
            self.task_kind = objective.kind
            self.output_dim = objective.output_dim
        self.n_classes = int(num_classes)

        # Avoid checkpoint bloat / pickle issues
        ignore_list = [
            "model_class",
            "loss_fct",
            "family",
            "objective",
            "cat_feature_info",
            "num_feature_info",
            "base_model_class",
            "gam_payload",
        ]
        self.save_hyperparameters(ignore=ignore_list)

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.optimizer_name = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_decay_steps = int(lr_decay_steps)
        self.lr_decay_factor = float(lr_decay_factor)
        self.lr_schedule = str(lr_schedule).lower()
        if self.lr_warmup_steps < 0 or self.lr_decay_steps < 0:
            raise ValueError("lr_warmup_steps and lr_decay_steps must be non-negative.")
        if not 0 < self.lr_decay_factor <= 1:
            raise ValueError("lr_decay_factor must lie in (0, 1].")
        if self.lr_schedule not in {
            "plateau",
            "inverse_sqrt",
            "cosine",
            "warmup_cosine",
            "none",
        }:
            raise ValueError(
                "lr_schedule must be 'plateau', 'inverse_sqrt', 'cosine', "
                "'warmup_cosine', or 'none'."
            )
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_mode = scheduler_mode

        self.model = model_class(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=self.output_dim,
            **kwargs,
        )

    def forward(self, num_features, cat_features):
        return self.model(num_features=num_features, cat_features=cat_features)

    @property
    def lss(self):
        return self.task_kind == "distributional"

    @property
    def family(self):
        if self.objective is None:
            return None
        return getattr(self.objective, "family", None)

    def compute_loss(self, predictions, y_true, sample_weight=None):
        return self.objective.compute_loss(
            predictions, y_true, sample_weight=sample_weight
        )

    def _log_task_metrics(
        self,
        prefix: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
        data_loss: torch.Tensor,
    ):
        if self.pretraining:
            return
        self.objective.log_metrics(self, prefix, preds, labels, data_loss)

    def _shared_step(self, batch, batch_idx, stage: str):
        if len(batch) == 4:
            cat_features, num_features, labels, offset = batch
            sample_weight = None
        else:
            cat_features, num_features, labels, offset, sample_weight = batch
            if torch.all(sample_weight == 1):
                sample_weight = None
        if self.pretraining:
            return self._masked_reconstruction_step(
                cat_features, num_features, stage=stage
            )

        result = self(num_features=num_features, cat_features=cat_features)
        preds = result["output"]
        if torch.any(offset != 0):
            if not self.objective.allows_offset:
                raise RuntimeError(
                    f"Per-sample offsets are not supported for {self.task_kind} "
                    "objectives."
                )
            preds = preds + offset
        data_loss = self.compute_loss(preds, labels, sample_weight=sample_weight)
        loss = data_loss
        penalty = harvest_penalties(result)
        if penalty is not None:
            loss = loss + penalty

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self._log_task_metrics(stage, preds, labels, data_loss)

        return loss

    def _masked_reconstruction_step(self, cat_features, num_features, *, stage):
        """Mirror NODE-GAM's masked-feature reconstruction objective."""
        ordered = list(num_features.items()) + list(cat_features.items())
        if not ordered:
            raise ValueError("Masked reconstruction requires at least one feature.")
        tensors = [tensor.float() for _, tensor in ordered]
        targets = torch.cat(tensors, dim=1)
        masks = torch.bernoulli(
            torch.full_like(targets, self.pretraining_ratio)
        )
        applied_masks = masks
        if self.pretraining_noise > 0:
            applied_masks = torch.bernoulli(
                (1.0 - self.pretraining_noise) * masks
            )
        masked = (1.0 - applied_masks) * targets

        masked_num = {}
        masked_cat = {}
        start = 0
        for name, tensor in num_features.items():
            width = tensor.shape[1]
            masked_num[name] = masked[:, start : start + width]
            start += width
        for name, tensor in cat_features.items():
            width = tensor.shape[1]
            masked_cat[name] = masked[:, start : start + width]
            start += width

        feature_masks = applied_masks if self.pretraining_feature_mask else None
        result = self.model(
            num_features=masked_num,
            cat_features=masked_cat,
            feature_masks=feature_masks,
        )
        outputs = result["output"]
        masks_per_row = masks.sum(dim=1, keepdim=True)
        masks_per_row[masks_per_row == 0] = 1
        loss = torch.mean((((outputs - targets) * masks) ** 2) / masks_per_row)
        penalty = harvest_penalties(result)
        if penalty is not None:
            loss = loss + penalty
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def on_train_batch_start(self, batch, batch_idx):
        callback = getattr(self.model, "temp_step_callback", None)
        if callback is not None:
            callback(self.global_step)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer_classes = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "adagrad": torch.optim.Adagrad,
            "sgd": torch.optim.SGD,
        }
        if isinstance(self.optimizer_name, str):
            name = self.optimizer_name.lower()
            if name == "qhadam":
                try:
                    from qhoptim.pyt import QHAdam
                except ImportError as error:
                    raise ImportError(
                        "optimizer='qhadam' requires the optional qhoptim package."
                    ) from error
                optimizer_class = QHAdam
            elif name in optimizer_classes:
                optimizer_class = optimizer_classes[name]
            else:
                raise ValueError(
                    "optimizer must be 'adam', 'adamw', 'adagrad', 'sgd', "
                    "'qhadam', or "
                    "an optimizer class."
                )
        else:
            optimizer_class = self.optimizer_name

        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs.setdefault("lr", self.lr)
        optimizer_kwargs.setdefault("weight_decay", self.weight_decay)
        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        if self.lr_schedule in {"cosine", "warmup_cosine"}:
            total_steps = self.lr_decay_steps
            if total_steps <= 0:
                total_steps = int(self.trainer.estimated_stepping_batches)
            total_steps = max(total_steps, 1)
            warmup_steps = self.lr_warmup_steps

            def cosine_multiplier(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=cosine_multiplier
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.lr_warmup_steps > 0 or self.lr_decay_steps > 0:
            def lr_multiplier(step):
                warmup = 1.0
                if self.lr_warmup_steps > 0:
                    warmup = min((step + 1) / self.lr_warmup_steps, 1.0)
                decay = 1.0
                if self.lr_decay_steps > 0:
                    decay = self.lr_decay_factor ** (step // self.lr_decay_steps)
                return warmup * decay

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lr_multiplier
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.lr_schedule == "none":
            return optimizer
        if self.lr_schedule == "inverse_sqrt":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: 1.0 / ((epoch + 1) ** 0.5)
                ),
                "interval": "epoch",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_mode,
                factor=self.lr_factor,
                patience=self.lr_patience,
            ),
            "monitor": self.scheduler_monitor,
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
