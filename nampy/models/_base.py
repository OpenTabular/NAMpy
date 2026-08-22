from __future__ import annotations

import inspect
import pickle
import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_sample_weight

from ..neural.data.datamodule import NAMpyDataModule
from ..neural.task import TaskModule
from ._data import prepare_fit_features, prepare_predict_features

EstimatorT = TypeVar("EstimatorT", bound="NeuralEstimatorBase")


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


def _preprocessor_defaults() -> dict[str, Any]:
    parameters = inspect.signature(Preprocessor.__init__).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _normalize_preprocessor_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if normalized.get("categorical_preprocessing") in ("one_hot", "one-hot"):
        normalized["categorical_preprocessing"] = "one-hot"
    if normalized.get("numerical_preprocessing") == "normalization":
        normalized["numerical_preprocessing"] = "minmax"
    return normalized


def seed_neural_runtime(seed: int) -> None:
    """Seed model construction and data-order RNGs from the public fit seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NeuralEstimatorBase(BaseEstimator):
    """Base class for shared NAMpy sklearn estimator behavior."""

    def fit(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        max_steps: int = -1,
        max_time=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        early_stopping_steps: int | None = None,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_decay_factor: float = 0.2,
        weight_decay: float | None = None,
        optimizer=None,
        optimizer_kwargs=None,
        warm_start=False,
        average_checkpoints: bool = False,
        n_last_checkpoints: int = 5,
        pretrain_epochs: int = 0,
        pretrain_max_steps: int = -1,
        pretraining_ratio: float = 0.15,
        pretraining_noise: float = 0.1,
        pretraining_feature_mask: bool = True,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs=None,
        offset=None,
        offset_val=None,
        sample_weight=None,
        sample_weight_val=None,
        class_weight=None,
        sampling_strategy=None,
        loss_fct=None,
        **trainer_kwargs,
    ):
        """Train the model on ``X``/``y``; optionally validate on a held-out set.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like
            The target values; task-specific shape and dtype rules apply.
        val_size : float, default=0.2
            Proportion for the automatic validation split when ``X_val`` is
            None; ignored otherwise.
        X_val, y_val : optional
            Explicit validation data; must be provided together.
        max_epochs : int, default=100
            Maximum number of Lightning training epochs. Architectures with a
            native optimizer, such as IGANN, use their model-specific stage
            count instead (``n_estimators`` for IGANN).
        max_steps : int, default=-1
            Optional global optimizer-step limit. ``-1`` leaves it unlimited.
        max_time : optional
            Lightning-compatible wall-clock limit, such as ``"00:30:00"``.
        random_state : int, default=101
            Seed for Python, NumPy, Torch, the automatic train/validation split,
            and data-loader ordering. Architectures may expose a more specific
            seed, such as GPNAM's ``rff_random_state``.
        batch_size : int, default=128
            Samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data each epoch.
        patience : int, default=15
            Early-stopping patience on the monitored metric.
        early_stopping_steps : int, optional
            When provided, replace epoch-check patience with optimizer-step
            distance since the last monitored improvement.
        monitor : str, default="val_loss"
            Metric monitored by early stopping and checkpoint selection.
        mode : str, default="min"
            Whether the monitored metric is minimized or maximized.
        lr, lr_patience, lr_factor, lr_schedule, weight_decay
            Optional fit-time overrides for the estimator's optimizer and
            LR-scheduler configuration. ``None`` uses the corresponding
            constructor/config value. ``lr_schedule`` accepts ``"plateau"``,
            ``"inverse_sqrt"``, or ``"none"``.
        lr_warmup_steps, lr_decay_steps, lr_decay_factor
            Optional step scheduler. Warmup is linear; decay multiplies the
            learning rate every ``lr_decay_steps`` optimizer steps.
        optimizer : str or optimizer class, optional
            Uses the architecture default when omitted. Supported names are
            ``"adam"``, ``"adamw"``, ``"adagrad"``, ``"sgd"``, and optional
            ``"qhadam"``; a compatible optimizer class is also accepted.
        optimizer_kwargs : dict, optional
            Extra optimizer constructor arguments.
        warm_start : bool or path, default=False
            Reuse the current fitted weights, or load weights from a Lightning
            checkpoint. This is weights-only; optimizer state is not resumed.
        average_checkpoints : bool, default=False
            Load the arithmetic mean of the latest epoch-end floating model
            states after fitting.
        n_last_checkpoints : int, default=5
            Number of recent epoch-end states retained for averaging.
        pretrain_epochs : int, default=0
            Run optional masked reconstruction before supervised fitting.
            Currently supported only by NodeGAM architectures.
        pretrain_max_steps, pretraining_ratio, pretraining_noise,
        pretraining_feature_mask
            Controls for NodeGAM reconstruction pretraining, mirroring the
            upstream ``pretrain_recon``/``pretrain_recon2`` objective.
        checkpoint_path : str, default="model_checkpoints"
            Root directory for per-fit checkpoint subdirectories.
        dataloader_kwargs : dict, optional
            Extra kwargs for the torch DataLoader.
        offset, offset_val : array-like, optional
            Per-sample additive offsets on the prediction (link) scale for
            the training and explicit-validation rows. Not supported for
            LSS tasks. Note that ``predict`` does NOT apply a stored offset;
            callers composing models must add offsets themselves.
        sample_weight, sample_weight_val : array-like, optional
            Non-negative per-row training and explicit-validation loss weights.
            Supported by regression, classification, and distributional objectives.
        class_weight : dict or "balanced", optional
            Class weights combined multiplicatively with ``sample_weight``.
        sampling_strategy : {None, "balanced"}, default=None
            For classifiers, optionally draw inverse-frequency balanced
            training batches. This is independent from loss weighting.
        loss_fct : callable, optional
            Custom loss ``fn(preds, targets)`` replacing the task default
            (regression tasks only; e.g. ``nn.PoissonNLLLoss(log_input=True)``
            for count responses composed on the log-link scale).
        **trainer_kwargs
            Additional kwargs for PyTorch Lightning's Trainer. These do not
            apply to architectures using native training.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        if "factor" in trainer_kwargs:
            raise TypeError(
                "fit() uses 'lr_factor'; the ambiguous 'factor' name is not "
                "forwarded to the Lightning Trainer."
            )
        seed_neural_runtime(random_state)
        if pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative.")
        if pretrain_max_steps != -1 and pretrain_max_steps < 1:
            raise ValueError("pretrain_max_steps must be -1 or at least 1.")
        X = prepare_fit_features(self, X)
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)

        raw_y = np.asarray(y)
        self._fit_loss_fct = loss_fct
        y, y_val, plan = self._build_training_plan(y, y_val)
        if class_weight is not None:
            if plan.datamodule_regression:
                raise ValueError("class_weight is only supported for classifiers.")
            class_weights = compute_sample_weight(class_weight, raw_y.reshape(-1))
            if sample_weight is None:
                sample_weight = class_weights
            else:
                sample_weight = np.asarray(sample_weight).reshape(-1) * class_weights
        if sampling_strategy is not None and plan.datamodule_regression:
            raise ValueError("sampling_strategy is only supported for classifiers.")
        lr = self.config.lr if lr is None else lr
        lr_patience = self.config.lr_patience if lr_patience is None else lr_patience
        lr_factor = self.config.lr_factor if lr_factor is None else lr_factor
        lr_schedule = (
            getattr(self.config, "lr_schedule", "plateau")
            if lr_schedule is None
            else lr_schedule
        )
        weight_decay = (
            self.config.weight_decay if weight_decay is None else weight_decay
        )
        optimizer = (
            getattr(self.config, "optimizer", "adam")
            if optimizer is None
            else optimizer
        )

        warm_start_state = None
        if warm_start:
            if isinstance(warm_start, (str, Path)):
                payload = torch.load(warm_start, map_location="cpu", weights_only=False)
                warm_start_state = payload.get("state_dict", payload)
            elif getattr(self, "model", None) is not None:
                warm_start_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                raise ValueError(
                    "warm_start=True requires an already fitted estimator or a "
                    "checkpoint path."
                )

        self._run_training(
            X,
            y,
            plan,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_time=max_time,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            patience=patience,
            early_stopping_steps=early_stopping_steps,
            monitor=monitor,
            mode=mode,
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
            warm_start_state=warm_start_state,
            average_checkpoints=average_checkpoints,
            n_last_checkpoints=n_last_checkpoints,
            pretrain_epochs=pretrain_epochs,
            pretrain_max_steps=pretrain_max_steps,
            pretraining_ratio=pretraining_ratio,
            pretraining_noise=pretraining_noise,
            pretraining_feature_mask=pretraining_feature_mask,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            trainer_kwargs=trainer_kwargs,
            offset=offset,
            offset_val=offset_val,
            sample_weight=sample_weight,
            sample_weight_val=sample_weight_val,
            sampling_strategy=sampling_strategy,
        )
        return self

    def _build_training_plan(self, y, y_val):
        """Return ``(y, y_val, TrainingPlan)`` for this task. Task-specific."""
        raise NotImplementedError

    def _prepare_architecture_config(
        self,
        config,
        *,
        train_num_features,
        train_cat_features,
        train_targets,
        val_num_features,
        val_cat_features,
        val_targets,
        objective,
        train_offset=None,
        train_sample_weight=None,
        random_state=0,
    ):
        """Resolve optional train-data-dependent architecture configuration.

        Most architectures return their constructor configuration unchanged.
        Estimator mixins can override this lifecycle hook for staged methods,
        such as reference-model interaction discovery, without moving that
        stateful workflow into the Torch forward architecture.
        """
        del (
            train_num_features,
            train_cat_features,
            train_targets,
            val_num_features,
            val_cat_features,
            val_targets,
            objective,
            train_offset,
            train_sample_weight,
            random_state,
        )
        return config

    def _run_training(
        self,
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
            bool(getattr(self.base_model, "supports_fixed_linear_regression", False))
            and getattr(plan.objective, "kind", None) == "regression"
            and getattr(plan.objective, "uses_default_loss", False)
            and str(getattr(self.config, "solver", "gradient")).lower() == "cg"
        )
        native_capable = bool(
            getattr(self.base_model, "supports_native_training", False)
        )
        native_selector = getattr(self.base_model, "uses_native_training", None)
        use_native_training = native_capable
        if native_selector is not None:
            use_native_training = bool(native_selector(plan.objective, self.config))
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
            raise ValueError(
                "warm_start is not supported by architecture-native training."
            )
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

        self.data_module = NAMpyDataModule(
            preprocessor=self.preprocessor,
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
        self.data_module.setup_data(
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

        train_cat, train_num = self.data_module.preprocess_tensors(
            self.data_module.X_train
        )
        val_cat, val_num = self.data_module.preprocess_tensors(self.data_module.X_val)

        def optional_tensor(value):
            return None if value is None else torch.as_tensor(value)

        architecture_config = self._prepare_architecture_config(
            self.config,
            train_num_features=train_num,
            train_cat_features=train_cat,
            train_targets=torch.as_tensor(self.data_module.y_train),
            val_num_features=val_num,
            val_cat_features=val_cat,
            val_targets=torch.as_tensor(self.data_module.y_val),
            objective=plan.objective,
            train_offset=optional_tensor(self.data_module.offset_train),
            train_sample_weight=optional_tensor(
                self.data_module.sample_weight_train
            ),
            random_state=random_state,
        )
        # Data-driven setup must not make final architecture initialization
        # depend on how much work its reference/selection stage performed.
        seed_neural_runtime(random_state)
        self.fitted_config_ = architecture_config

        pretrained_state = None
        self.pretrained_keys_ = ()
        if pretrain_epochs > 0:
            architecture_name = getattr(self, "architecture_name", None)
            if architecture_name is None:
                supports_pretraining = getattr(
                    self.base_model, "supports_masked_pretraining", False
                )
            else:
                from ..neural.registry import get_architecture

                supports_pretraining = get_architecture(
                    architecture_name
                ).supports("masked_pretraining")
            if not supports_pretraining:
                raise ValueError(
                    f"{self.base_model.__name__} does not support masked reconstruction "
                    "pretraining."
                )
            if not 0 <= pretraining_ratio <= 1:
                raise ValueError("pretraining_ratio must lie between 0 and 1.")
            if not 0 <= pretraining_noise <= 1:
                raise ValueError("pretraining_noise must lie between 0 and 1.")

            input_width = sum(
                info["dimension"]
                for info in self.data_module.num_feature_info.values()
            ) + sum(
                info["dimension"]
                for info in self.data_module.cat_feature_info.values()
            )
            pretraining_config = replace(architecture_config, interaction_degree=1)
            pretraining_model = TaskModule(
                model_class=self.base_model,
                config=pretraining_config,
                cat_feature_info=self.data_module.cat_feature_info,
                num_feature_info=self.data_module.num_feature_info,
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
            pretrainer.fit(pretraining_model, self.data_module)
            pretrained_state = {
                key: value.detach().cpu().clone()
                for key, value in pretraining_model.state_dict().items()
            }

        self.model = TaskModule(
            model_class=self.base_model,
            config=architecture_config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
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

        initialize = getattr(
            self.model.model, "initialize_from_training_data", None
        )
        if initialize is not None:
            with torch.no_grad():
                initialize(train_num, train_cat)
        if native_capable and not use_native_training:
            prepare_gradient_training = getattr(
                self.model.model, "prepare_gradient_training", None
            )
            if prepare_gradient_training is None:
                raise RuntimeError(
                    f"{self.base_model.__name__} selected generic training but did "
                    "not implement prepare_gradient_training()."
                )
            prepare_gradient_training()

        if pretrained_state is not None:
            target_state = self.model.state_dict()
            compatible = {
                key: value
                for key, value in pretrained_state.items()
                if key in target_state and value.shape == target_state[key].shape
            }
            target_state.update(compatible)
            self.model.load_state_dict(target_state)
            self.pretrained_keys_ = tuple(sorted(compatible))

        if warm_start_state is not None:
            try:
                self.model.load_state_dict(warm_start_state, strict=True)
            except RuntimeError as error:
                raise ValueError(
                    "Warm-start checkpoint is incompatible with the current model."
                ) from error

        if use_native_training:
            self.native_training_info_ = dict(
                self.model.model.fit_native(
                    train_num_features=train_num,
                    train_cat_features=train_cat,
                    train_targets=torch.as_tensor(self.data_module.y_train),
                    val_num_features=val_num,
                    val_cat_features=val_cat,
                    val_targets=torch.as_tensor(self.data_module.y_val),
                    objective=plan.objective,
                    train_offset=optional_tensor(self.data_module.offset_train),
                    val_offset=optional_tensor(self.data_module.offset_val),
                    train_sample_weight=optional_tensor(
                        self.data_module.sample_weight_train
                    ),
                    val_sample_weight=optional_tensor(
                        self.data_module.sample_weight_val
                    ),
                    random_state=random_state,
                )
            )
            self.best_model_path_ = None
            self.averaged_checkpoints_ = 0
            self.linear_solver_info_ = None
            self._record_model_metadata()
            return

        if use_fixed_solver:
            from ..neural.linear_solver import solve_fixed_linear_regression

            self.linear_solver_info_ = solve_fixed_linear_regression(
                self.model.model,
                num_features=train_num,
                cat_features=train_cat,
                targets=self.data_module.y_train,
                sample_weight=self.data_module.sample_weight_train,
                offset=self.data_module.offset_train,
            )
            self.best_model_path_ = None
            self.averaged_checkpoints_ = 0
            self._record_model_metadata()
            return

        early_stop_callback, checkpoint_callback = build_callbacks(
            type(self).__name__,
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
        trainer.fit(self.model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        self.best_model_path_ = best_model_path or None
        if averaging_callback is not None and averaging_callback.states:
            self.averaged_checkpoints_ = len(averaging_callback.states)
            self.model.load_state_dict(averaging_callback.averaged_state_dict())
        elif best_model_path:
            self.averaged_checkpoints_ = 0
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.averaged_checkpoints_ = 0
        self.linear_solver_info_ = None
        self.native_training_info_ = None
        self._record_model_metadata()

    def _record_model_metadata(self) -> None:
        """Expose backend-independent complexity and fitted architecture state."""
        parameters = tuple(self.model.parameters())
        self.n_parameters_total_ = int(
            sum(parameter.numel() for parameter in parameters)
        )
        self.n_parameters_trainable_ = int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        )
        architecture = self.model.model
        for name in getattr(architecture, "estimator_fitted_attributes", ()):
            setattr(self, name, getattr(architecture, name))

    def _predict_raw(self, X, *, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Run inference on prepared features and return the forward-output dict.

        ``X`` must already have passed through the estimator's feature
        preparation. The model's train/eval state is restored afterwards.
        """
        from ..neural.contracts import is_penalty_key

        cat_tensor_dict, num_tensor_dict = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat_tensor_dict.values()) + tuple(num_tensor_dict.values())
        if not tensors:
            raise ValueError("Prediction requires at least one transformed feature.")
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        device = next(self.model.parameters()).device

        was_training = self.model.training
        self.model.eval()
        collected: dict[str, list[torch.Tensor]] = {}
        constants: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for start in range(0, n_rows, batch_size):
                    stop = min(start + batch_size, n_rows)
                    cat_batch = {
                        key: tensor[start:stop].to(device)
                        for key, tensor in cat_tensor_dict.items()
                    }
                    num_batch = {
                        key: tensor[start:stop].to(device)
                        for key, tensor in num_tensor_dict.items()
                    }
                    result = self.model(
                        num_features=num_batch, cat_features=cat_batch
                    )
                    for key, value in result.items():
                        is_batched = (
                            key != "intercept"
                            and not is_penalty_key(key)
                            and value.ndim > 0
                            and value.shape[0] == stop - start
                        )
                        if is_batched:
                            collected.setdefault(key, []).append(value)
                        elif key not in constants:
                            constants[key] = value
        finally:
            self.model.train(was_training)

        merged = {
            key: torch.cat(values, dim=0) for key, values in collected.items()
        }
        merged.update(constants)
        return merged

    def _predict(self, X, *, batch_size: int | None = None):
        """Run inference and return the raw forward-output dict."""
        if getattr(self, "model", None) is None or (
            getattr(self, "data_module", None) is None
        ):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        X = prepare_predict_features(self, X)
        return self._predict_raw(X, batch_size=batch_size)

    def basis_transform(self, X, *, batch_size: int | None = None):
        """Return an optional architecture's fixed design matrix."""
        if getattr(self, "model", None) is None or self.data_module is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        architecture = self.model.model
        transform = getattr(architecture, "linear_design", None)
        if transform is None:
            raise NotImplementedError(
                f"{type(architecture).__name__} does not expose a fixed basis."
            )
        X = prepare_predict_features(self, X)
        cat, num = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat.values()) + tuple(num.values())
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        device = next(architecture.parameters()).device
        blocks = []
        with torch.no_grad():
            for start in range(0, n_rows, int(batch_size)):
                stop = min(start + int(batch_size), n_rows)
                blocks.append(
                    transform(
                        {key: value[start:stop].to(device) for key, value in num.items()},
                        {key: value[start:stop].to(device) for key, value in cat.items()},
                    ).cpu()
                )
        return torch.cat(blocks).numpy()

    def local_term_importance(
        self,
        X,
        *,
        target: int = 0,
        top_k: int = 10,
        batch_size: int | None = None,
    ):
        """Return architecture-specific local source-term expansions.

        This is currently implemented by SPAM. Other architectures fail
        explicitly rather than substituting generic additive importance for a
        model-specific polynomial expansion.
        """
        if getattr(self, "model", None) is None or self.data_module is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        architecture = self.model.model
        importance = getattr(architecture, "local_term_importance", None)
        if importance is None:
            raise NotImplementedError(
                f"{type(architecture).__name__} has no local term expansion."
            )
        X = prepare_predict_features(self, X)
        cat, num = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat.values()) + tuple(num.values())
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        device = next(architecture.parameters()).device
        rows = []
        was_training = architecture.training
        architecture.eval()
        try:
            with torch.no_grad():
                for start in range(0, n_rows, int(batch_size)):
                    stop = min(start + int(batch_size), n_rows)
                    rows.extend(
                        importance(
                            {
                                key: value[start:stop].to(device)
                                for key, value in num.items()
                            },
                            {
                                key: value[start:stop].to(device)
                                for key, value in cat.items()
                            },
                            target=target,
                            top_k=top_k,
                        )
                    )
        finally:
            architecture.train(was_training)
        return rows

    def basis_metadata(self) -> dict:
        """Return fitted fixed-basis metadata when the architecture provides it."""
        if getattr(self, "model", None) is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        metadata = getattr(self.model.model, "basis_metadata", None)
        if metadata is None:
            raise NotImplementedError(
                f"{type(self.model.model).__name__} has no fixed-basis metadata."
            )
        return metadata()

    def model_complexity(self) -> dict[str, int]:
        """Return generic and optional architecture-specific complexity counts."""
        if not hasattr(self, "n_parameters_total_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        result = {
            "total_parameters": self.n_parameters_total_,
            "trainable_parameters": self.n_parameters_trainable_,
        }
        architecture_metadata = getattr(
            self.model.model, "complexity_metadata", None
        )
        if architecture_metadata is not None:
            result.update(architecture_metadata())
        return result

    def training_history(self) -> dict:
        """Return architecture-native fit history when one is available."""
        if getattr(self, "model", None) is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        history = getattr(self.model.model, "training_history_", None)
        if history is None:
            raise NotImplementedError(
                f"{type(self.model.model).__name__} has no native training history."
            )
        return dict(history)

    def _split_output_components(self, pred_dict):
        """Split a forward-output dict into (terms, intercept) numpy parts."""
        from ..neural.contracts import is_penalty_key

        terms: dict[str, Any] = {}
        intercept: float | Any = 0.0
        for key, value in pred_dict.items():
            if key == "output" or is_penalty_key(key):
                continue
            array = value.detach().cpu().numpy()
            if key == "intercept":
                intercept = float(array.item()) if array.size == 1 else array
            else:
                terms[key] = array
        return terms, intercept

    def _maybe_center_components(
        self,
        prediction,
        *,
        center,
        reference_X=None,
        reference_weight=None,
    ):
        if not center and reference_X is None:
            return prediction
        from ..explanations import center_additive_prediction

        reference = None
        if reference_X is not None:
            reference = self.predict_components(reference_X)
        return center_additive_prediction(
            prediction,
            reference=reference,
            sample_weight=reference_weight,
        )

    def center_components(self, X, *, reference_X=None, reference_weight=None):
        """Return contributions centered on ``X`` or a reference dataset."""
        return self.predict_components(
            X,
            center=True,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )

    def _plot_series_labels(self, n_series: int):
        """Labels for the per-output plot lines; None means a single line."""
        return None

    def plot(self, X, y_true, feature_name=None, plot_interactions=False):
        """Legacy density-style view of per-feature effects.

        Draws each feature's contribution curve over a density-shaded
        background with the (centered) targets scattered behind it. Kept for
        back-compat; prefer :meth:`plot_terms` for term-contribution curves
        via the shared renderer, and :meth:`plot_interactions` for interaction
        heatmaps and conditioned higher-order slices.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        y_true : np.ndarray
            True target values shown as scatter behind the curves.
        feature_name : str, optional
            Specific feature to plot; all numerical features when None.
        plot_interactions : bool, optional
            Whether to also plot fitted feature interactions.
        """
        from ._plotting import plot_feature_effects

        plot_feature_effects(
            self,
            X,
            y_true,
            feature_name=feature_name,
            plot_interactions=plot_interactions,
        )

    def plot_interactions(self, X):
        """Plot fitted interaction terms.

        Renders a binned heatmap for pairwise terms and conditioned observed
        heatmap slices for higher-order ``":"``-keyed terms.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        """
        from ._plotting import plot_interaction_heatmaps

        plot_interaction_heatmaps(self, X)

    def plot_terms(
        self,
        X,
        *,
        center=False,
        reference_X=None,
        reference_weight=None,
        rug=None,
        pages=0,
        figsize=None,
    ):
        """Render 1-d term-contribution curves via the shared renderer.

        Builds the same prepared-data schema the GAM plotting pipeline uses,
        so GAM and neural term plots share one renderer. Only single-column
        numeric main effects are drawn; interaction terms use
        :meth:`plot_interactions`.
        """
        from ..plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(
            X,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )
        prepared = prepared_from_contributions(X, components.terms)
        return render_term_plots(prepared, rug=rug, pages=pages, figsize=figsize)

    def explain_terms(
        self,
        X,
        *,
        max_bins: int = 64,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return a binned, backend-neutral additive-term explanation table."""
        from ..explanations import explain_additive_prediction

        return explain_additive_prediction(
            X,
            self.predict_components(
                X,
                center=center,
                reference_X=reference_X,
                reference_weight=reference_weight,
            ),
            max_bins=max_bins,
        )

    def term_importance(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return mean absolute link-scale contribution by additive term."""
        from ..explanations import term_importance_table

        return term_importance_table(
            self.predict_components(
                X,
                center=center,
                reference_X=reference_X,
                reference_weight=reference_weight,
            )
        )

    def interaction_importance(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return the interaction-only subset of :meth:`term_importance`."""
        table = self.term_importance(
            X,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )
        return table.loc[table["term_type"] == "interaction"].reset_index(drop=True)

    def save_model(self, path: str | Path) -> Path:
        """Persist this estimator in the versioned NAMpy pickle format.

        Pickle artifacts are executable Python objects and must only be loaded
        from trusted sources. The envelope makes incompatible formats fail
        explicitly instead of producing obscure unpickle errors.
        """
        destination = Path(path)
        payload = {
            "format": "nampy-estimator",
            "version": 1,
            "estimator_class": type(self).__name__,
            "estimator": self,
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle)
        return destination

    @classmethod
    def load_model(cls: type[EstimatorT], path: str | Path) -> EstimatorT:
        """Load a version-1 estimator artifact written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        if (
            not isinstance(loaded, dict)
            or loaded.get("format") != "nampy-estimator"
            or loaded.get("version") != 1
        ):
            raise ValueError(f"{source} is not a supported NAMpy estimator artifact.")
        estimator = loaded.get("estimator")
        if not isinstance(estimator, cls):
            raise TypeError(
                f"{source} contains {type(estimator).__name__}, not {cls.__name__}."
            )
        return estimator

    def _initialize_estimator_parameters(self, config_class, kwargs):
        config_names = set(getattr(config_class, "__dataclass_fields__", {}))
        preprocessor_defaults = _preprocessor_defaults()
        preprocessor_names = set(preprocessor_defaults)

        flat_kwargs = {}
        nested_preprocessor_kwargs = {}
        for name, value in kwargs.items():
            if name.startswith("preprocessor__"):
                nested_preprocessor_kwargs[name.split("__", 1)[1]] = value
            else:
                flat_kwargs[name] = value

        unknown_flat = set(flat_kwargs) - config_names - preprocessor_names
        unknown_nested = set(nested_preprocessor_kwargs) - preprocessor_names
        if unknown_flat or unknown_nested:
            unknown = sorted(unknown_flat | unknown_nested)
            valid = sorted(config_names | preprocessor_names)
            raise TypeError(
                f"Unexpected parameter(s) {unknown} for {self.__class__.__name__}. "
                f"Valid parameters are {valid}."
            )

        config_inputs = {
            name: value for name, value in flat_kwargs.items() if name in config_names
        }
        explicit_preprocessor = {
            name: value
            for name, value in flat_kwargs.items()
            if name in preprocessor_names
        }
        explicit_preprocessor.update(nested_preprocessor_kwargs)
        explicit_preprocessor = _normalize_preprocessor_params(explicit_preprocessor)
        self._config_param_names = tuple(sorted(config_names))
        self._preprocessor_param_names = tuple(sorted(preprocessor_names))
        self.config = config_class(**config_inputs)
        self.config_kwargs = {
            name: getattr(self.config, name) for name in self._config_param_names
        }
        self._preprocessor_kwargs = dict(preprocessor_defaults)
        self._preprocessor_kwargs.update(explicit_preprocessor)
        self._provided_preprocessor_kwargs = explicit_preprocessor
        self._rebuild_preprocessor()

    def _rebuild_preprocessor(self):
        self.preprocessor = Preprocessor(**self._preprocessor_kwargs)

    def get_params(self, deep=True):
        params = dict(self.config_kwargs)
        for name, value in self._preprocessor_kwargs.items():
            if name not in params:
                params[name] = value
            elif params[name] != value:
                params[f"preprocessor__{name}"] = value

        if deep:
            params.update(
                {
                    f"preprocessor__{name}": value
                    for name, value in self._preprocessor_kwargs.items()
                }
            )
        return params

    def set_params(self, **parameters):
        config_updates = {}
        preprocessor_updates = {}

        for name, value in parameters.items():
            if name.startswith("preprocessor__"):
                preprocessor_name = name.split("__", 1)[1]
                if preprocessor_name not in self._preprocessor_param_names:
                    raise ValueError(
                        f"Invalid parameter {name!r} for {self.__class__.__name__}."
                    )
                preprocessor_updates[preprocessor_name] = value
                continue

            owns_config = name in self._config_param_names
            owns_preprocessor = name in self._preprocessor_param_names
            if not owns_config and not owns_preprocessor:
                valid = sorted(
                    set(self._config_param_names) | set(self._preprocessor_param_names)
                )
                raise ValueError(
                    f"Invalid parameter {name!r} for {self.__class__.__name__}. "
                    f"Valid parameters are {valid}."
                )
            if owns_config:
                config_updates[name] = value
            if owns_preprocessor:
                preprocessor_updates[name] = value

        for name, value in config_updates.items():
            setattr(self.config, name, value)
            self.config_kwargs[name] = value

        if preprocessor_updates:
            self._preprocessor_kwargs.update(
                _normalize_preprocessor_params(preprocessor_updates)
            )
            self._rebuild_preprocessor()

        return self
