"""Neural fit orchestration, callbacks and checkpoint selection."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .data.datamodule import NAMpyDataModule
from .task import TaskModule


def _training_tensor(value, *, device=None):
    """Convert training inputs without sharing read-only NumPy storage."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, device=device)


@dataclass
class TrainingPlan:
    """Task-specific wiring for the shared neural training engine.

    Attributes
    ----------
    objective : NeuralObjective
        Output, target, loss, and metric semantics forwarded to
        :class:`TaskModule` independently of the architecture.
    stratify : array-like or None
        Stratification labels for the automatic train/validation split.
    """

    objective: Any
    stratify: Any = None

    @property
    def datamodule_regression(self) -> bool:
        """Derive the data-label representation from the objective."""
        return bool(self.objective.datamodule_regression)


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
        self.states = deque(maxlen=int(n_last_checkpoints))

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


def seed_neural_runtime(seed: int) -> None:
    """Seed model construction and data-order RNGs from the public fit seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    max_steps,
    max_time,
    random_state,
    batch_size,
    shuffle,
    patience,
    early_stopping_steps,
    monitor,
    mode,
    lr,
    lr_patience,
    lr_factor,
    lr_schedule,
    lr_warmup_steps,
    lr_decay_steps,
    lr_decay_factor,
    weight_decay,
    optimizer,
    optimizer_kwargs,
    warm_start_state,
    average_checkpoints,
    n_last_checkpoints,
    pretrain_epochs,
    pretrain_max_steps,
    pretraining_ratio,
    pretraining_noise,
    pretraining_feature_mask,
    checkpoint_path,
    dataloader_kwargs,
    trainer_kwargs,
    offset=None,
    offset_val=None,
    sample_weight=None,
    sample_weight_val=None,
    sampling_strategy=None,
) -> None:
    """Fit this estimator in place: data module, TaskModule, Trainer, reload."""
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    use_fixed_solver = (
        bool(getattr(estimator.base_model, "supports_fixed_linear_regression", False))
        and getattr(plan.objective, "kind", None) == "regression"
        and getattr(plan.objective, "uses_default_loss", False)
        and str(getattr(estimator.config, "solver", "gradient")).lower() == "cg"
    )
    native_capable = bool(
        getattr(estimator.base_model, "supports_native_training", False)
    )
    native_selector = getattr(estimator.base_model, "uses_native_training", None)
    use_native_training = native_capable
    if native_selector is not None:
        use_native_training = bool(native_selector(plan.objective, estimator.config))
    if use_fixed_solver and use_native_training:
        raise RuntimeError(
            "An architecture cannot use fixed-design and native training together."
        )
    if use_native_training and getattr(plan.objective, "kind", None) == "regression":
        if not getattr(plan.objective, "uses_default_loss", False):
            raise NotImplementedError(
                "Architecture-native regression currently requires the default "
                "squared-error objective."
            )
    if use_fixed_solver and pretrain_epochs > 0:
        raise ValueError("Fixed-design regression cannot use masked pretraining.")
    if use_fixed_solver and warm_start_state is not None:
        raise ValueError(
            "warm_start is not meaningful for an exact fixed-design solve."
        )
    if use_native_training and pretrain_epochs > 0:
        raise ValueError("Architecture-native training cannot use masked pretraining.")
    if use_native_training and warm_start_state is not None:
        raise ValueError("warm_start is not supported by architecture-native training.")
    if use_native_training and average_checkpoints:
        raise ValueError(
            "Checkpoint averaging is not supported by architecture-native training."
        )

    setup_X_val, setup_y_val = X_val, y_val
    setup_offset_val, setup_weight_val = offset_val, sample_weight_val
    if use_fixed_solver and X_val is None:
        # Exact fitting does not need early-stopping rows. Fit preprocessing
        # and coefficients on every supplied training row.
        setup_X_val, setup_y_val = X, y
        setup_offset_val, setup_weight_val = offset, sample_weight

    estimator.data_module = NAMpyDataModule(
        preprocessor=estimator.preprocessor,
        batch_size=batch_size,
        shuffle=shuffle,
        X_val=setup_X_val,
        y_val=setup_y_val,
        val_size=val_size,
        random_state=random_state,
        regression=plan.datamodule_regression,
        sampling_strategy=sampling_strategy,
        **dataloader_kwargs,
    )
    estimator.data_module.setup_data(
        X,
        y,
        X_val=setup_X_val,
        y_val=setup_y_val,
        val_size=val_size,
        random_state=random_state,
        stratify=plan.stratify,
        offset=offset,
        offset_val=setup_offset_val,
        sample_weight=sample_weight,
        sample_weight_val=setup_weight_val,
    )

    train_cat, train_num = estimator.data_module.preprocess_tensors(
        estimator.data_module.X_train
    )
    val_cat, val_num = estimator.data_module.preprocess_tensors(
        estimator.data_module.X_val
    )

    prepare_config = getattr(estimator.base_model, "prepare_training_config", None)
    architecture_config = estimator.config
    if prepare_config is not None:
        architecture_config, fitted_attributes = prepare_config(
            estimator.config,
            train_num_features=train_num,
            train_cat_features=train_cat,
            train_targets=_training_tensor(estimator.data_module.y_train),
            val_num_features=val_num,
            val_cat_features=val_cat,
            val_targets=_training_tensor(estimator.data_module.y_val),
            objective=plan.objective,
            train_offset=_training_tensor(estimator.data_module.offset_train),
            train_sample_weight=_training_tensor(
                estimator.data_module.sample_weight_train
            ),
            random_state=random_state,
        )
        for name, value in fitted_attributes.items():
            setattr(estimator, name, value)
    # Data-driven setup must not make final architecture initialization
    # depend on how much work its reference/selection stage performed.
    seed_neural_runtime(random_state)
    estimator.fitted_config_ = architecture_config

    pretrained_state = None
    estimator.pretrained_keys_ = ()
    if pretrain_epochs > 0:
        architecture_name = getattr(estimator, "architecture_name", None)
        if architecture_name is None:
            supports_pretraining = getattr(
                estimator.base_model, "supports_masked_pretraining", False
            )
        else:
            from ..neural.registry import get_architecture

            supports_pretraining = get_architecture(architecture_name).supports(
                "masked_pretraining"
            )
        if not supports_pretraining:
            raise ValueError(
                f"{estimator.base_model.__name__} does not support masked reconstruction "
                "pretraining."
            )
        if not 0 <= pretraining_ratio <= 1:
            raise ValueError("pretraining_ratio must lie between 0 and 1.")
        if not 0 <= pretraining_noise <= 1:
            raise ValueError("pretraining_noise must lie between 0 and 1.")

        input_width = sum(
            info["dimension"]
            for info in estimator.data_module.num_feature_info.values()
        ) + sum(
            info["dimension"]
            for info in estimator.data_module.cat_feature_info.values()
        )
        pretraining_config = replace(architecture_config, interaction_degree=1)
        pretraining_model = TaskModule(
            model_class=estimator.base_model,
            config=pretraining_config,
            cat_feature_info=estimator.data_module.cat_feature_info,
            num_feature_info=estimator.data_module.num_feature_info,
            num_classes=input_width,
            pretraining=True,
            pretraining_ratio=pretraining_ratio,
            pretraining_noise=pretraining_noise,
            pretraining_feature_mask=pretraining_feature_mask,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_factor=lr_decay_factor,
            weight_decay=weight_decay,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_monitor="val_loss",
            scheduler_mode="min",
        )
        pretrain_options = {
            key: value
            for key, value in trainer_kwargs.items()
            if key not in {"callbacks", "max_steps", "max_time"}
        }
        pretrain_options.setdefault("logger", False)
        pretrain_options.setdefault("enable_checkpointing", False)
        pretrain_options.setdefault("enable_model_summary", False)
        if pretrain_max_steps != -1:
            pretrain_options["max_steps"] = pretrain_max_steps
        pretrainer = pl.Trainer(
            max_epochs=pretrain_epochs,
            **pretrain_options,
        )
        pretrainer.fit(pretraining_model, estimator.data_module)
        pretrained_state = {
            key: value.detach().cpu().clone()
            for key, value in pretraining_model.state_dict().items()
        }

    estimator.model = TaskModule(
        model_class=estimator.base_model,
        config=architecture_config,
        cat_feature_info=estimator.data_module.cat_feature_info,
        num_feature_info=estimator.data_module.num_feature_info,
        lr=lr,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_schedule=lr_schedule,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_factor=lr_decay_factor,
        weight_decay=weight_decay,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_monitor=monitor,
        scheduler_mode=mode,
        objective=plan.objective,
    )

    initialize = getattr(estimator.model.model, "initialize_from_training_data", None)
    if initialize is not None:
        with torch.no_grad():
            initialize(train_num, train_cat)
    if native_capable and not use_native_training:
        prepare_gradient_training = getattr(
            estimator.model.model, "prepare_gradient_training", None
        )
        if prepare_gradient_training is None:
            raise RuntimeError(
                f"{estimator.base_model.__name__} selected generic training but did "
                "not implement prepare_gradient_training()."
            )
        prepare_gradient_training()

    if pretrained_state is not None:
        target_state = estimator.model.state_dict()
        compatible = {
            key: value
            for key, value in pretrained_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target_state.update(compatible)
        estimator.model.load_state_dict(target_state)
        estimator.pretrained_keys_ = tuple(sorted(compatible))

    if warm_start_state is not None:
        try:
            estimator.model.load_state_dict(warm_start_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Warm-start checkpoint is incompatible with the current model."
            ) from error

    if use_native_training:
        estimator.native_training_info_ = dict(
            estimator.model.model.fit_native(
                train_num_features=train_num,
                train_cat_features=train_cat,
                train_targets=_training_tensor(estimator.data_module.y_train),
                val_num_features=val_num,
                val_cat_features=val_cat,
                val_targets=_training_tensor(estimator.data_module.y_val),
                objective=plan.objective,
                train_offset=_training_tensor(estimator.data_module.offset_train),
                val_offset=_training_tensor(estimator.data_module.offset_val),
                train_sample_weight=_training_tensor(
                    estimator.data_module.sample_weight_train
                ),
                val_sample_weight=_training_tensor(
                    estimator.data_module.sample_weight_val
                ),
                random_state=random_state,
            )
        )
        estimator.best_model_path_ = None
        estimator.averaged_checkpoints_ = 0
        estimator.linear_solver_info_ = None
        estimator._record_model_metadata()
        return

    if use_fixed_solver:
        from ..neural.linear_solver import solve_fixed_linear_regression

        estimator.linear_solver_info_ = solve_fixed_linear_regression(
            estimator.model.model,
            num_features=train_num,
            cat_features=train_cat,
            targets=estimator.data_module.y_train,
            sample_weight=estimator.data_module.sample_weight_train,
            offset=estimator.data_module.offset_train,
        )
        estimator.best_model_path_ = None
        estimator.averaged_checkpoints_ = 0
        estimator._record_model_metadata()
        return

    early_stop_callback, checkpoint_callback = build_callbacks(
        type(estimator).__name__,
        monitor=monitor,
        mode=mode,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    callbacks = [checkpoint_callback]
    if early_stopping_steps is None:
        callbacks.insert(0, early_stop_callback)
    else:
        callbacks.insert(
            0,
            StepEarlyStopping(
                monitor=monitor,
                mode=mode,
                patience_steps=early_stopping_steps,
            ),
        )
    averaging_callback = None
    if average_checkpoints:
        averaging_callback = RecentStateAveraging(n_last_checkpoints)
        callbacks.append(averaging_callback)

    trainer_options = dict(trainer_kwargs)
    user_callbacks = trainer_options.pop("callbacks", None)
    if user_callbacks is not None:
        if isinstance(user_callbacks, (list, tuple)):
            callbacks.extend(user_callbacks)
        else:
            callbacks.append(user_callbacks)
    if max_steps != -1:
        trainer_options["max_steps"] = max_steps
    if max_time is not None:
        trainer_options["max_time"] = max_time

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        **trainer_options,
    )
    trainer.fit(estimator.model, estimator.data_module)

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
