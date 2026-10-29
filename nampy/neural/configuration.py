"""Constructor configuration with one owner for each neural control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectiveConfig:
    family: str | None = None
    distributional_kwargs: dict | None = None
    loss_fct: Any = None
    class_weight: Any = None
    sampling_strategy: str | None = None

    def __post_init__(self):
        if self.sampling_strategy not in {None, "balanced"}:
            raise ValueError("sampling_strategy must be None or 'balanced'.")


@dataclass(frozen=True)
class SurvivalObjectiveConfig(ObjectiveConfig):
    """Survival likelihood settings, independent of architecture and training."""

    cuts: tuple[float, ...] | None = None
    likelihood: str = "piecewise_exponential"

    def __post_init__(self):
        super().__post_init__()
        from ..survival.contracts import validate_cuts

        if self.likelihood == "piecewise_exponential":
            object.__setattr__(self, "cuts", tuple(validate_cuts((0., 1.) if self.cuts is None else self.cuts)))
        elif self.likelihood in {"weibull", "lognormal", "cox"}:
            if self.cuts is not None:
                raise ValueError("This survival likelihood does not use cuts; omit them.")
        else:
            raise NotImplementedError("Supported survival likelihoods are piecewise_exponential, weibull, lognormal and cox.")
        if any(value is not None for value in (
            self.family, self.distributional_kwargs, self.loss_fct,
            self.class_weight, self.sampling_strategy,
        )):
            raise NotImplementedError("Survival objectives do not support family overrides, custom losses or class balancing.")


@dataclass(frozen=True, init=False)
class PreprocessingConfig:
    """Validated PreTab constructor parameters, kept separate from model controls."""

    parameters: dict

    def __init__(self, **parameters):
        from inspect import signature

        from pretab.preprocessor import Preprocessor

        signature(Preprocessor).bind_partial(**parameters)
        object.__setattr__(self, "parameters", dict(parameters))


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 101
    val_size: float = 0.2
    batch_size: int = 128
    shuffle: bool = True
    dataloader_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GradientTrainingConfig(TrainingConfig):
    max_epochs: int = 100
    max_steps: int = -1
    max_time: Any = None
    patience: int = 15
    early_stopping_steps: int | None = None
    monitor: str = "val_loss"
    mode: str = "min"
    lr: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.1
    lr_schedule: str = "plateau"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_decay_factor: float = 0.2
    weight_decay: float = 1e-6
    optimizer: Any = "adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    warm_start: bool | str = False
    average_checkpoints: bool = False
    n_last_checkpoints: int = 5
    pretrain_epochs: int = 0
    pretrain_max_steps: int = -1
    pretraining_ratio: float = 0.15
    pretraining_noise: float = 0.1
    pretraining_feature_mask: bool = True
    checkpoint_path: str | None = None
    trainer_kwargs: dict = field(default_factory=dict)

    SCHEDULES = frozenset(
        {"plateau", "inverse_sqrt", "cosine", "warmup_cosine", "none"}
    )

    def __post_init__(self):
        if self.lr_schedule not in self.SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {sorted(self.SCHEDULES)}.")
        if self.lr_warmup_steps < 0 or self.lr_decay_steps < 0:
            raise ValueError("lr_warmup_steps and lr_decay_steps must be non-negative.")
        if not 0 < self.lr_decay_factor <= 1:
            raise ValueError("lr_decay_factor must lie in (0, 1].")
        if self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative.")
        if self.pretrain_max_steps != -1 and self.pretrain_max_steps < 1:
            raise ValueError("pretrain_max_steps must be -1 or at least 1.")
        from dataclasses import fields

        overlap = set(self.trainer_kwargs) & {field.name for field in fields(self)}
        if overlap:
            raise ValueError(
                f"Set training controls on GradientTrainingConfig, not trainer_kwargs: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class FixedDesignTrainingConfig(TrainingConfig):
    ridge: float = 0.05
    cg_rtol: float = 1e-6
    cg_max_iter: int | None = None

    def __post_init__(self):
        if self.ridge < 0 or self.cg_rtol <= 0:
            raise ValueError("ridge must be nonnegative and cg_rtol positive.")


@dataclass(frozen=True)
class NativeTrainingConfig(TrainingConfig):
    """Common controls for an architecture-owned training algorithm."""


@dataclass(frozen=True)
class IGANNTrainingConfig(NativeTrainingConfig):
    """IGANN boosting controls, independent of gradient optimizers."""

    boost_rate: float = 0.1
    init_reg: float = 1.0
    elm_alpha: float = 1.0
    early_stopping: int = 50
    sparse: int = 0
    device: str = "cpu"
    clip_predictions: float = 100.0


@dataclass(frozen=True)
class SemiStructuredTrainingConfig(NativeTrainingConfig):
    """Full-cohort joint optimization with fitted training-space projection."""

    epochs: int = 100
    lr: float = .01
    patience: int = 20
    neural_weight_decay: float = 0.
    smoothing_params: Any = None

    def __post_init__(self):
        if self.epochs < 1 or self.lr <= 0 or self.patience < 1 or self.neural_weight_decay < 0:
            raise ValueError("Composition training needs positive epochs/lr/patience and nonnegative decay.")


GRADIENT_DEFAULTS = {
    "DefaultTreeNAMConfig": {"lr": 0.001},
    "DefaultGPNAMConfig": {
        "lr_schedule": "inverse_sqrt",
        "weight_decay": 0.0,
        "lr": 0.01,
    },
    "DefaultNBMSPAMConfig": {"weight_decay": 0.0, "lr": 0.001},
    "DefaultIGANNConfig": {"lr_schedule": "none", "weight_decay": 0.0, "lr": 0.1},
    "DefaultSIANConfig": {"optimizer": "adagrad", "weight_decay": 0.0, "lr": 0.005},
    "DefaultSPAMConfig": {"weight_decay": 0.0, "lr": 0.001},
}
