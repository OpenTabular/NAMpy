"""Resolve training support from architecture, objective and route configuration."""

import numpy as np

from ..configuration import (
    GRADIENT_DEFAULTS,
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    NativeTrainingConfig,
)


def resolve_training(model, config_class, objective, training, task, y):
    kind = task
    if task == "classifier":
        kind = "binary" if np.unique(np.asarray(y)).size == 2 else "multiclass"
    elif task == "regressor":
        kind = "regression"
    native = bool(getattr(model, "supports_native_training", False))
    native_supported = native and kind in getattr(model, "native_objective_kinds", ())
    fixed_supported = (
        bool(getattr(model, "supports_fixed_linear_regression", False))
        and kind == "regression"
        and objective.loss_fct is None
    )
    if training is None:
        if native_supported and objective.loss_fct is None:
            training = getattr(model, "native_training_config", NativeTrainingConfig)()
        elif fixed_supported:
            training = FixedDesignTrainingConfig()
        else:
            training = GradientTrainingConfig(
                **GRADIENT_DEFAULTS.get(config_class.__name__, {})
            )
    if isinstance(training, NativeTrainingConfig):
        if not native_supported or objective.loss_fct is not None:
            raise NotImplementedError(
                "Native training requires a supported architecture/task and its default loss."
            )
        expected = getattr(model, "native_training_config", NativeTrainingConfig)
        if not isinstance(training, expected):
            raise TypeError(
                f"{model.__name__} native training requires {expected.__name__}."
            )
    elif isinstance(training, FixedDesignTrainingConfig):
        if not fixed_supported:
            raise NotImplementedError(
                "Fixed-design training requires a fixed linear architecture and default regression loss."
            )
    elif not isinstance(training, GradientTrainingConfig):
        raise TypeError("Unknown training configuration.")
    if not isinstance(training, GradientTrainingConfig) and (
        objective.sampling_strategy is not None
        or {"sampler", "batch_sampler"} & training.dataloader_kwargs.keys()
    ):
        raise NotImplementedError(
            "Resampling requires GradientTrainingConfig; native and fixed-design "
            "routes operate on the supplied training rows directly."
        )
    return training
