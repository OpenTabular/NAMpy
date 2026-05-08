import inspect
from typing import Any, Type

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class TaskModel(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        num_classes=1,  # semantic class count for classification, output dim for regression/LSS
        lss=False,
        family=None,
        loss_fct: Any = None,
        **kwargs,
    ):
        super().__init__()

        self.lss = bool(lss)
        self.family = family
        self.loss_fct = loss_fct

        # Keep task semantics separate from model output width
        self.n_classes = int(num_classes)

        if self.lss:
            self.task_kind = "lss"
            self.output_dim = int(num_classes)  # usually family.param_count
        else:
            if self.n_classes == 1:
                self.task_kind = "regression"
                self.output_dim = 1
                if self.loss_fct is None:
                    self.loss_fct = nn.MSELoss()
            elif self.n_classes == 2:
                self.task_kind = "binary"
                self.output_dim = 1  # BCE-with-logits style
                if self.loss_fct is None:
                    self.loss_fct = nn.BCEWithLogitsLoss()
                self.acc = torchmetrics.Accuracy(task="binary")
                # Optional: keep AUROC/precision, but log on_epoch only (see below)
                self.auroc = torchmetrics.AUROC(task="binary")
                self.precision = torchmetrics.Precision(task="binary")
            else:
                self.task_kind = "multiclass"
                self.output_dim = self.n_classes
                if self.loss_fct is None:
                    self.loss_fct = nn.CrossEntropyLoss()
                self.acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.n_classes
                )
                self.auroc = torchmetrics.AUROC(
                    task="multiclass", num_classes=self.n_classes
                )
                self.precision = torchmetrics.Precision(
                    task="multiclass", num_classes=self.n_classes
                )

        # Avoid checkpoint bloat / pickle issues
        ignore_list = [
            "model_class",
            "loss_fct",
            "family",
            "cat_feature_info",
            "num_feature_info",
        ]
        self.save_hyperparameters(ignore=ignore_list)

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        model_kwargs = dict(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=self.output_dim,
            **kwargs,
        )
        if "family" in inspect.signature(model_class).parameters:
            model_kwargs["family"] = family
        self.model = model_class(**model_kwargs)
        self._model_accepts_return_terms = (
            "return_terms" in inspect.signature(self.model.forward).parameters
        )

    def forward(self, num_features, cat_features, return_terms=True):
        kwargs = {"num_features": num_features, "cat_features": cat_features}
        if self._model_accepts_return_terms:
            kwargs["return_terms"] = return_terms
        return self.model(**kwargs)

    def _prepare_supervised_targets(self, preds: torch.Tensor, y_true: torch.Tensor):
        """Normalize target shapes/dtypes for regression/binary/multiclass."""
        if self.task_kind == "regression":
            y = y_true.to(dtype=preds.dtype)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            return preds, y

        if self.task_kind == "binary":
            # BCEWithLogitsLoss expects same shape/dtype as preds
            y = y_true.to(dtype=preds.dtype)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            if y.shape != preds.shape:
                y = y.view_as(preds)
            return preds, y

        if self.task_kind == "multiclass":
            # CrossEntropyLoss expects logits [N, C], targets [N] long
            y = y_true.long().view(-1)
            return preds, y

        raise RuntimeError(f"Unexpected task_kind={self.task_kind!r}")

    def _prepare_lss_targets(self, y_true: torch.Tensor):
        # Keep multi-parameter / multivariate targets intact.
        # Only squeeze the dummy second dim when labels are shape [N, 1].
        y = y_true
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]

        # Let family opt into integer targets (e.g. categorical)
        target_dtype = getattr(self.family, "target_dtype", None)
        if target_dtype is not None:
            y = y.to(dtype=target_dtype)
        return y

    def compute_loss(self, predictions, y_true):
        if self.lss:
            y = self._prepare_lss_targets(y_true)
            return self.family.compute_loss(predictions, y)

        preds, y = self._prepare_supervised_targets(predictions, y_true)
        return self.loss_fct(preds, y)

    def _log_task_metrics(self, prefix: str, preds: torch.Tensor, labels: torch.Tensor):
        if self.lss:
            return

        if self.task_kind == "binary":
            # For torchmetrics binary classification, logits are accepted in many versions,
            # but using probs explicitly avoids version-specific behavior.
            probs = torch.sigmoid(preds).view(-1)
            y = labels.view(-1).long()
            self.log(
                f"{prefix}_acc",
                self.acc(probs, y),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                f"{prefix}_auroc",
                self.auroc(probs, y),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{prefix}_precision",
                self.precision(probs, y),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        elif self.task_kind == "multiclass":
            logits = preds
            y = labels.view(-1).long()
            self.log(
                f"{prefix}_acc",
                self.acc(logits, y),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            probs = torch.softmax(logits, dim=1)
            self.log(
                f"{prefix}_auroc",
                self.auroc(probs, y),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{prefix}_precision",
                self.precision(logits, y),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    def _shared_step(self, batch, batch_idx, stage: str):
        cat_features, num_features, labels = batch
        result = self(
            num_features=num_features, cat_features=cat_features, return_terms=False
        )
        preds = result["output"]
        loss = self.compute_loss(preds, labels)
        for key, value in result.items():
            if key.endswith("_penalty") or key.endswith("_regularizer"):
                loss = loss + value

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self._log_task_metrics(stage, preds, labels)

        if stage == "test" and (not self.lss) and self.task_kind == "regression":
            self.log(
                "test_rmse",
                torch.sqrt(loss),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def _step_temperature_schedulers(self):
        if hasattr(self.model, "step_temperature_schedulers"):
            self.model.step_temperature_schedulers(int(self.global_step))

    def training_step(self, batch, batch_idx):
        self._step_temperature_schedulers()
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
