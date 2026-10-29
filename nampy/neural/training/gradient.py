"""Lightning callbacks and checkpoint averaging for gradient execution."""

from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ..task import TaskModule


def build_callbacks(name: str, *, monitor, mode, patience, checkpoint_path):
    """Build the EarlyStopping/ModelCheckpoint pair for one fit.

    Both callbacks honour the user's ``monitor``/``mode``. Each fit writes into
    its own subdirectory so concurrent or successive fits never clobber each
    other's checkpoints.
    """
    early_stop_callback = EarlyStopping(
        monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
    )
    run_dir = Path(checkpoint_path) / f"{name}-{uuid4().hex[:8]}"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        dirpath=str(run_dir),
        filename="best_model",
    )
    return early_stop_callback, checkpoint_callback


def average_state_dicts(state_dicts):
    """Average floating checkpoint tensors and retain the newest discrete state."""
    if not state_dicts:
        raise ValueError("At least one state dict is required for averaging.")
    keys = tuple(state_dicts[0])
    if any(tuple(state) != keys for state in state_dicts[1:]):
        raise ValueError("All state dicts must contain the same ordered keys.")

    averaged = {}
    for key in keys:
        tensors = [state[key] for state in state_dicts]
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulation_dtype = (
                torch.complex128 if tensors[0].is_complex() else torch.float64
            )
            accumulator = tensors[0].to(dtype=accumulation_dtype).clone()
            for tensor in tensors[1:]:
                accumulator.add_(tensor.to(dtype=accumulation_dtype))
            averaged[key] = accumulator.div_(len(tensors)).to(dtype=tensors[0].dtype)
        else:
            averaged[key] = tensors[-1].clone()
    return averaged


class RecentStateAveraging(pl.Callback):
    """Keep the latest epoch-end model states for post-fit averaging."""

    def __init__(self, n_last_checkpoints: int):
        if n_last_checkpoints < 1:
            raise ValueError("n_last_checkpoints must be at least 1.")
        super().__init__()
        self.states: deque[dict[str, torch.Tensor]] = deque(
            maxlen=int(n_last_checkpoints)
        )

    def on_train_epoch_end(self, trainer, pl_module):
        del trainer
        self.states.append(
            {
                key: value.detach().cpu().clone()
                for key, value in pl_module.state_dict().items()
            }
        )

    def averaged_state_dict(self):
        return average_state_dicts(list(self.states))


class StepEarlyStopping(pl.Callback):
    """Stop after a monitored metric has not improved for N optimizer steps."""

    def __init__(self, *, monitor, mode, patience_steps, min_delta=0.0):
        if patience_steps < 1:
            raise ValueError("patience_steps must be at least 1.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience_steps = int(patience_steps)
        self.min_delta = float(min_delta)
        self.best = None
        self.best_step = 0

    def on_validation_end(self, trainer, pl_module):
        del pl_module
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            raise RuntimeError(
                f"StepEarlyStopping could not find monitored metric {self.monitor!r}."
            )
        current = float(value.detach().cpu() if torch.is_tensor(value) else value)
        improved = self.best is None
        if self.best is not None and self.mode == "min":
            improved = current < self.best - self.min_delta
        elif self.best is not None and self.mode == "max":
            improved = current > self.best + self.min_delta
        if improved:
            self.best = current
            self.best_step = int(trainer.global_step)
        elif int(trainer.global_step) - self.best_step >= self.patience_steps:
            trainer.should_stop = True


def pretrain(estimator, architecture_config, training):
    """Optional masked reconstruction on the already fitted training schema."""
    if training.pretrain_epochs <= 0:
        return None
    architecture_name = getattr(estimator, "architecture_name", None)
    if architecture_name is None:
        supports_pretraining = getattr(
            estimator.base_model, "supports_masked_pretraining", False
        )
    else:
        from ..registry import get_architecture

        supports_pretraining = get_architecture(architecture_name).supports(
            "masked_pretraining"
        )
    if not supports_pretraining:
        raise ValueError(
            f"{estimator.base_model.__name__} does not support masked reconstruction "
            "pretraining."
        )
    if not 0 <= training.pretraining_ratio <= 1:
        raise ValueError("pretraining_ratio must lie between 0 and 1.")
    if not 0 <= training.pretraining_noise <= 1:
        raise ValueError("pretraining_noise must lie between 0 and 1.")

    input_width = sum(
        info["dimension"] for info in estimator.data_module.num_feature_info.values()
    ) + sum(
        info["dimension"] for info in estimator.data_module.cat_feature_info.values()
    )
    pretraining_config = replace(architecture_config, interaction_degree=1)
    pretraining_model = TaskModule(
        model_class=estimator.base_model,
        config=pretraining_config,
        cat_feature_info=estimator.data_module.cat_feature_info,
        num_feature_info=estimator.data_module.num_feature_info,
        reconstruction_width=input_width,
        pretraining=True,
        pretraining_ratio=training.pretraining_ratio,
        pretraining_noise=training.pretraining_noise,
        pretraining_feature_mask=training.pretraining_feature_mask,
        training=training,
    )
    pretrain_options = {
        key: value
        for key, value in training.trainer_kwargs.items()
        if key not in {"callbacks", "max_steps", "max_time"}
    }
    pretrain_options.setdefault("logger", False)
    pretrain_options.setdefault("enable_checkpointing", False)
    pretrain_options.setdefault("enable_model_summary", False)
    if training.pretrain_max_steps != -1:
        pretrain_options["max_steps"] = training.pretrain_max_steps
    pretrainer = pl.Trainer(
        max_epochs=training.pretrain_epochs,
        **pretrain_options,
    )
    pretrainer.fit(pretraining_model, estimator.data_module)
    pretrained_state = {
        key: value.detach().cpu().clone()
        for key, value in pretraining_model.state_dict().items()
    }

    return pretrained_state


def fit_gradient(estimator, training, *, fisher=None):
    """Clean temporary checkpoints on success/failure; retain explicitly requested paths."""
    directory = (
        TemporaryDirectory(prefix="nampy-fit-")
        if training.checkpoint_path is None
        else nullcontext(training.checkpoint_path)
    )
    with directory as checkpoint_root:
        _fit_gradient(
            estimator,
            training,
            checkpoint_root,
            **({"fisher": fisher} if fisher is not None else {}),
        )
        if training.checkpoint_path is None:
            estimator.best_model_path_ = None


def _fit_gradient(estimator, training, checkpoint_root, *, fisher=None):
    """Run Lightning and select weights, retaining scalar diagnostics only."""
    early_stop_callback, checkpoint_callback = build_callbacks(
        type(estimator).__name__,
        monitor=training.monitor,
        mode=training.mode,
        patience=training.patience,
        checkpoint_path=checkpoint_root,
    )

    callbacks = [checkpoint_callback]
    if training.early_stopping_steps is None:
        callbacks.insert(0, early_stop_callback)
    else:
        callbacks.insert(
            0,
            StepEarlyStopping(
                monitor=training.monitor,
                mode=training.mode,
                patience_steps=training.early_stopping_steps,
            ),
        )
    averaging_callback = None
    if training.average_checkpoints:
        averaging_callback = RecentStateAveraging(training.n_last_checkpoints)
        callbacks.append(averaging_callback)

    if fisher is not None and fisher.config.every_n_epochs is not None:
        callbacks.append(fisher)

    trainer_options = dict(training.trainer_kwargs)
    trainer_options.setdefault("default_root_dir", str(checkpoint_root))
    user_callbacks = trainer_options.pop("callbacks", None)
    if user_callbacks is not None:
        if isinstance(user_callbacks, (list, tuple)):
            callbacks.extend(user_callbacks)
        else:
            callbacks.append(user_callbacks)
    if training.max_steps != -1:
        trainer_options["max_steps"] = training.max_steps
    if training.max_time is not None:
        trainer_options["max_time"] = training.max_time

    trainer = pl.Trainer(
        max_epochs=training.max_epochs,
        callbacks=callbacks,
        **trainer_options,
    )
    trainer.fit(estimator.model, estimator.data_module)
    estimator.training_completion_ = {
        "finished": bool(trainer.state.finished),
        "interrupted": bool(trainer.interrupted),
    }

    # Retain lightweight training-resource metadata without keeping the
    # Trainer object (which would complicate estimator persistence).  The
    # public attributes support reproducible benchmark accounting for
    # epochs, optimizer steps, and the final monitored values.
    estimator.n_epochs_ = int(trainer.current_epoch)
    estimator.n_optimizer_steps_ = int(trainer.global_step)
    estimator.final_training_metrics_ = {
        str(key): float(value.detach().cpu() if torch.is_tensor(value) else value)
        for key, value in trainer.callback_metrics.items()
        if np.asarray(value.detach().cpu() if torch.is_tensor(value) else value).size
        == 1
    }

    best_model_path = checkpoint_callback.best_model_path
    estimator.best_model_path_ = best_model_path or None
    if averaging_callback is not None and averaging_callback.states:
        estimator.averaged_checkpoints_ = len(averaging_callback.states)
        estimator.model.load_state_dict(averaging_callback.averaged_state_dict())
    elif best_model_path:
        estimator.averaged_checkpoints_ = 0
        checkpoint = torch.load(best_model_path, weights_only=False)
        estimator.model.load_state_dict(checkpoint["state_dict"])
    else:
        estimator.averaged_checkpoints_ = 0
    estimator.linear_solver_info_ = None
    estimator.native_training_info_ = None
    estimator._record_model_metadata()
