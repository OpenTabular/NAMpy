"""Shared training/inference engine behind the sklearn-style neural wrappers.

The three public wrappers (regressor, classifier, LSS) differ only in target
preparation and output post-processing; everything between — data module,
TaskModule, callbacks, Trainer, best-checkpoint reload, and raw inference —
lives here exactly once. Wrappers describe their task with a
:class:`TrainingPlan` and delegate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.exceptions import NotFittedError

from ..data.datamodule import NAMpyDataModule
from .task_module import TaskModule


@dataclass
class TrainingPlan:
    """Task-specific wiring for the shared neural training engine.

    Attributes
    ----------
    datamodule_regression : bool
        Label dtype switch for :class:`NAMpyDataModule` (float32 vs long).
    taskmodel_kwargs : dict
        Task wiring forwarded to :class:`TaskModule` (``num_classes``,
        ``task`` or ``lss``/``family``).
    stratify : array-like or None
        Stratification labels for the automatic train/validation split.
    """

    datamodule_regression: bool
    taskmodel_kwargs: dict[str, Any] = field(default_factory=dict)
    stratify: Any = None
    passthrough: dict[str, Any] | None = None
    passthrough_val: dict[str, Any] | None = None


def build_callbacks(estimator, *, monitor, mode, patience, checkpoint_path):
    """Build the EarlyStopping/ModelCheckpoint pair for one fit.

    Both callbacks honour the user's ``monitor``/``mode``. Each fit writes into
    its own subdirectory so concurrent or successive fits never clobber each
    other's checkpoints.
    """
    early_stop_callback = EarlyStopping(
        monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
    )
    run_dir = Path(checkpoint_path) / f"{type(estimator).__name__}-{uuid4().hex[:8]}"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        dirpath=str(run_dir),
        filename="best_model",
    )
    return early_stop_callback, checkpoint_callback


def run_training(
    estimator,
    X,
    y,
    plan: TrainingPlan,
    *,
    val_size,
    X_val,
    y_val,
    max_epochs,
    random_state,
    batch_size,
    shuffle,
    patience,
    monitor,
    mode,
    lr,
    lr_patience,
    factor,
    weight_decay,
    checkpoint_path,
    dataloader_kwargs,
    trainer_kwargs,
    offset=None,
    offset_val=None,
) -> None:
    """Fit ``estimator`` in place: data module, TaskModule, Trainer, reload."""
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    estimator.data_module = NAMpyDataModule(
        preprocessor=estimator.preprocessor,
        batch_size=batch_size,
        shuffle=shuffle,
        X_val=X_val,
        y_val=y_val,
        val_size=val_size,
        random_state=random_state,
        regression=plan.datamodule_regression,
        **dataloader_kwargs,
    )
    estimator.data_module.setup_data(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        val_size=val_size,
        random_state=random_state,
        stratify=plan.stratify,
        offset=offset,
        offset_val=offset_val,
        passthrough_arrays=plan.passthrough,
        passthrough_arrays_val=plan.passthrough_val,
    )

    estimator.model = TaskModule(
        model_class=estimator.base_model,
        config=estimator.config,
        cat_feature_info=estimator.data_module.cat_feature_info,
        num_feature_info=estimator.data_module.num_feature_info,
        lr=lr,
        lr_patience=lr_patience,
        lr_factor=factor,
        weight_decay=weight_decay,
        **plan.taskmodel_kwargs,
    )

    early_stop_callback, checkpoint_callback = build_callbacks(
        estimator,
        monitor=monitor,
        mode=mode,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        **trainer_kwargs,
    )
    trainer.fit(estimator.model, estimator.data_module)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        checkpoint = torch.load(best_model_path, weights_only=False)
        estimator.model.load_state_dict(checkpoint["state_dict"])


def predict_raw(estimator, X) -> dict[str, torch.Tensor]:
    """Run inference on prepared features and return the forward-output dict.

    ``X`` must already have passed through the estimator's feature
    preparation. The model's train/eval state is restored afterwards.
    """
    if getattr(estimator, "model", None) is None or (
        getattr(estimator, "data_module", None) is None
    ):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this method."
        )

    cat_tensor_dict, num_tensor_dict = estimator.data_module.preprocess_test_data(X)

    device = next(estimator.model.parameters()).device
    cat_tensor_dict = {
        key: tensor.to(device) for key, tensor in cat_tensor_dict.items()
    }
    num_tensor_dict = {
        key: tensor.to(device) for key, tensor in num_tensor_dict.items()
    }

    was_training = estimator.model.training
    estimator.model.eval()
    try:
        with torch.no_grad():
            return estimator.model(
                num_features=num_tensor_dict, cat_features=cat_tensor_dict
            )
    finally:
        estimator.model.train(was_training)
