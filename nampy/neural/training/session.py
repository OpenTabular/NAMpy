"""Transactional neural fitting; publish state only after successful execution."""

from copy import deepcopy

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from ...models._data import prepare_fit_features, prepare_predict_features
from .common import seed_neural_runtime
from .engine import run_training


def fit_estimator(
    estimator,
    X,
    y,
    *,
    X_val=None,
    y_val=None,
    offset=None,
    offset_val=None,
    sample_weight=None,
    sample_weight_val=None,
    warm_start_state=None,
):
    estimator.training_config_ = estimator._resolved_training(y)
    training = deepcopy(estimator.training_config_)
    objective_config = estimator.objective
    estimator._prepare_fit(y)
    estimator.fisher_offset_required_ = offset is not None
    seed_neural_runtime(training.random_state)
    X = prepare_fit_features(estimator, X)
    if (X_val is None) ^ (y_val is None):
        raise ValueError("X_val and y_val must be provided together.")
    if X_val is not None:
        X_val = prepare_predict_features(estimator, X_val)

    raw_y = np.asarray(y) if objective_config.class_weight is not None else None
    y, y_val, plan = estimator._build_training_plan(y, y_val)
    if objective_config.class_weight is not None:
        if plan.datamodule_regression:
            raise ValueError("class_weight is only supported for classifiers.")
        class_weights = compute_sample_weight(
            objective_config.class_weight, raw_y.reshape(-1)
        )
        if sample_weight is None:
            sample_weight = class_weights
        else:
            sample_weight = np.asarray(sample_weight).reshape(-1) * class_weights
    if objective_config.sampling_strategy is not None and plan.datamodule_regression:
        raise ValueError("sampling_strategy is only supported for classifiers.")
    run_training(
        estimator,
        X,
        y,
        plan,
        training=training,
        X_val=X_val,
        y_val=y_val,
        offset=offset,
        offset_val=offset_val,
        sample_weight=sample_weight,
        sample_weight_val=sample_weight_val,
        warm_start_state=warm_start_state,
        sampling_strategy=objective_config.sampling_strategy,
    )
    estimator._finish_fit()
    completion = getattr(estimator, "training_completion_", None)
    if completion is not None and (
        not completion["finished"] or completion["interrupted"]
    ):
        raise RuntimeError("Neural training did not complete successfully.")
    if completion is not None and getattr(estimator, "n_optimizer_steps_", 0) < 1:
        raise RuntimeError("Training finished without an optimizer step.")
    if any(
        not np.isfinite(value)
        for key, value in getattr(estimator, "final_training_metrics_", {}).items()
        if key in {"train_loss", "train_loss_epoch", "val_loss"}
    ):
        raise RuntimeError("Neural training reported a nonfinite loss.")
    from ..fitted import FittedPreprocessing, NeuralPredictionModule

    data = estimator.data_module
    estimator.fitted_preprocessing_ = FittedPreprocessing(
        data.preprocessor, data.cat_feature_info, data.num_feature_info
    )
    estimator.training_rows_ = len(data.X_train)
    estimator.validation_rows_ = len(data.X_val)
    estimator.model = NeuralPredictionModule(
        estimator.model.model, estimator.model.objective
    )
    del estimator.data_module
    del estimator.preprocessor
    return estimator
