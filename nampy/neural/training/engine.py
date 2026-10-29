"""Prepare training data once and dispatch the selected execution route."""

from copy import deepcopy

import torch

from ..configuration import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    NativeTrainingConfig,
)
from ..data.datamodule import NAMpyDataModule
from ..fitted import NeuralPredictionModule
from ..task import TaskModule
from .common import TrainingPlan, _training_tensor, seed_neural_runtime
from .gradient import fit_gradient, pretrain
from .native import fit_fixed_design, fit_native


def run_training(
    estimator,
    X,
    y,
    plan: TrainingPlan,
    *,
    training,
    X_val=None,
    y_val=None,
    offset=None,
    offset_val=None,
    sample_weight=None,
    sample_weight_val=None,
    warm_start_state=None,
    sampling_strategy=None,
):
    """Prepare data and architecture, then dispatch a validated fitting route."""
    use_fixed_solver = isinstance(training, FixedDesignTrainingConfig)
    use_native_training = isinstance(training, NativeTrainingConfig)
    native_capable = bool(
        getattr(estimator.base_model, "supports_native_training", False)
    )
    if warm_start_state is not None and not isinstance(
        training, GradientTrainingConfig
    ):
        raise ValueError("Warm starts require gradient training.")

    setup_X_val, setup_y_val = X_val, y_val
    setup_offset_val, setup_weight_val = offset_val, sample_weight_val
    if use_fixed_solver and X_val is None:
        # Exact fitting does not need early-stopping rows. Fit preprocessing
        # and coefficients on every supplied training row.
        setup_X_val, setup_y_val = X, y
        setup_offset_val, setup_weight_val = offset, sample_weight

    estimator.data_module = NAMpyDataModule(
        preprocessor=estimator.preprocessor,
        batch_size=training.batch_size,
        shuffle=training.shuffle,
        X_val=setup_X_val,
        y_val=setup_y_val,
        val_size=training.val_size,
        random_state=training.random_state,
        regression=plan.datamodule_regression,
        sampling_strategy=sampling_strategy,
        preprocessing_targets=plan.objective.preprocessing_targets,
        target_dtype=getattr(plan.objective, "target_dtype", None),
        **training.dataloader_kwargs,
    )
    estimator.data_module.setup_data(
        X,
        y,
        X_val=setup_X_val,
        y_val=setup_y_val,
        val_size=training.val_size,
        random_state=training.random_state,
        stratify=plan.stratify,
        offset=offset,
        offset_val=setup_offset_val,
        sample_weight=sample_weight,
        sample_weight_val=setup_weight_val,
    )

    plan.objective.configure_data(estimator.data_module)

    train_cat, train_num = estimator.data_module.preprocess_tensors(
        estimator.data_module.X_train
    )
    val_cat, val_num = estimator.data_module.preprocess_tensors(
        estimator.data_module.X_val
    )

    prepare_config = getattr(estimator.base_model, "prepare_training_config", None)
    architecture_config = deepcopy(estimator.architecture_config)
    if prepare_config is not None:
        architecture_config, fitted_attributes = prepare_config(
            architecture_config,
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
            random_state=training.random_state,
        )
        for name, value in fitted_attributes.items():
            setattr(estimator, name, value)
    # Data-driven setup must not make final architecture initialization
    # depend on how much work its reference/selection stage performed.
    seed_neural_runtime(training.random_state)
    estimator.fitted_config_ = architecture_config

    estimator.pretrained_keys_ = ()
    pretrained_state = (
        pretrain(estimator, architecture_config, training)
        if isinstance(training, GradientTrainingConfig)
        else None
    )

    if isinstance(training, GradientTrainingConfig):
        estimator.model = TaskModule(
            model_class=estimator.base_model,
            config=architecture_config,
            cat_feature_info=estimator.data_module.cat_feature_info,
            num_feature_info=estimator.data_module.num_feature_info,
            training=training,
            objective=plan.objective,
        )
    else:
        architecture = estimator.base_model(
            config=architecture_config,
            cat_feature_info=estimator.data_module.cat_feature_info,
            num_feature_info=estimator.data_module.num_feature_info,
            num_classes=plan.objective.output_dim,
            **({"training": training} if native_capable else {}),
        )
        estimator.model = NeuralPredictionModule(architecture, plan.objective)

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
        fit_native(estimator, plan, train_num, train_cat, val_num, val_cat, training)
    elif use_fixed_solver:
        fit_fixed_design(estimator, train_num, train_cat, training)
    else:
        fit_gradient(estimator, training)
