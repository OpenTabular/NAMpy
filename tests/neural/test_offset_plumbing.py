"""Per-sample offset plumbing through dataset, datamodule, and TaskModule."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pretab.preprocessor import Preprocessor

from nampy.models.linreg import LinRegRegressor
from nampy.neural.configuration import GradientTrainingConfig, PreprocessingConfig
from nampy.neural.data.datamodule import NAMpyDataModule
from nampy.neural.objectives import DistributionObjective, RegressionObjective
from nampy.neural.task import TaskModule


class _ZeroOutputModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, num_features, cat_features):
        batch_size = next(iter(num_features.values())).shape[0]
        output = next(iter(num_features.values())).new_zeros(
            (batch_size, self.num_classes)
        )
        return {"output": output}


def _task_config():
    return SimpleNamespace(lr=1e-4, lr_patience=2, weight_decay=0.0, lr_factor=0.5)


def _regression_task_model():
    task_model = TaskModule(model_class=_ZeroOutputModel, config=_task_config(), cat_feature_info={}, num_feature_info={'x': {'dimension': 1}}, objective=RegressionObjective(1), training=GradientTrainingConfig())
    task_model.log = lambda *args, **kwargs: None
    return task_model


def test_zero_offset_batches_reproduce_offsetless_loss():
    task_model = _regression_task_model()
    features = {"x": torch.tensor([[0.0], [1.0], [2.0]])}
    labels = torch.ones((3, 1))

    loss = task_model.test_step(({}, features, labels, torch.zeros((3, 1))), 0)
    # Zero model output vs labels of one: plain MSE of 1.0, exactly as before
    # the offset channel existed.
    assert loss == pytest.approx(1.0)


def test_offset_shifts_regression_loss_exactly():
    task_model = _regression_task_model()
    features = {"x": torch.tensor([[0.0], [1.0], [2.0]])}
    labels = torch.ones((3, 1))

    loss = task_model.test_step(({}, features, labels, torch.ones((3, 1))), 0)
    # output(0) + offset(1) == label(1) -> zero loss.
    assert loss == pytest.approx(0.0)


def test_lss_offsets_require_exact_predictor_width():
    family = SimpleNamespace(
        param_count=2,
        compute_loss=lambda preds, y, reduction="none": preds.sum() * 0.0,
    )
    task_model = TaskModule(model_class=_ZeroOutputModel, config=_task_config(), cat_feature_info={}, num_feature_info={'x': {'dimension': 1}}, objective=DistributionObjective(family), training=GradientTrainingConfig())
    task_model.log = lambda *args, **kwargs: None
    features = {"x": torch.tensor([[0.0], [1.0]])}
    labels = torch.ones((2, 1))

    # Zero offsets pass through.
    task_model.test_step(({}, features, labels, torch.zeros((2, 1))), 0)

    task_model.test_step(({}, features, labels, torch.ones((2, 2))), 0)
    with pytest.raises(ValueError, match="no predictor broadcasting"):
        task_model.test_step(({}, features, labels, torch.ones((2, 1))), 0)


@pytest.mark.parametrize("longitudinal", [False, True])
@pytest.mark.parametrize("column_offset", [False, True])
def test_piecewise_subject_offsets_match_interval_likelihood(longitudinal, column_offset):
    from nampy.neural.survival import (
        LongitudinalPiecewiseObjective,
        PiecewiseExponentialObjective,
    )

    exposure = np.array([[1.0, 0.5], [0.5, 0.0], [1.0, 1.0], [1.0, 0.2]])
    events = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    allocation = np.full(4, 0.5) if longitudinal else np.ones(4)
    targets = np.column_stack([exposure, events])
    if longitudinal:
        targets = np.column_stack([targets, allocation])
    train_offset = np.array([-0.2, 0.1, 0.3, -0.1])
    val_offset = np.array([0.4, -0.3])
    train_weight = np.array([1.0, 2.0, 0.5, 0.0])
    data = SimpleNamespace(
        y_train=targets,
        y_val=targets[:2],
        offset_train=train_offset[:, None] if column_offset else train_offset,
        offset_val=val_offset[:, None] if column_offset else val_offset,
        sample_weight_train=train_weight,
        sample_weight_val=np.ones(2),
    )
    cls = LongitudinalPiecewiseObjective if longitudinal else PiecewiseExponentialObjective
    objective = cls(2)
    objective.configure_data(data)
    task_model = TaskModule(
        model_class=_ZeroOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        objective=objective,
        training=GradientTrainingConfig(),
    )
    task_model.log = lambda *args, **kwargs: None

    for stage, offsets, weights in (
        ("train", train_offset, train_weight),
        ("val", val_offset, np.ones(2)),
    ):
        expanded = getattr(data, f"offset_{stage}")
        np.testing.assert_array_equal(expanded, np.repeat(offsets[:, None], 2, axis=1))
        n = len(offsets)
        batch = (
            {}, {"x": torch.zeros(n, 1)}, torch.tensor(targets[:n]),
            torch.tensor(expanded), torch.tensor(weights),
        )
        actual = task_model._shared_step(batch, 0, stage)
        row_losses = (
            exposure[:n] * np.exp(offsets[:, None]) - events[:n] * offsets[:, None]
        ).sum(axis=1)
        expected = np.sum(weights * row_losses) / np.sum(weights * allocation[:n])
        assert actual.item() == pytest.approx(expected, rel=2e-7)


def test_datamodule_splits_offset_alongside_features():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=100), "z": rng.normal(size=100)})
    y = rng.normal(size=100)
    offset = X["x"].to_numpy().copy()

    data_module = NAMpyDataModule(
        preprocessor=Preprocessor(numerical_preprocessing="standardization"),
        batch_size=32,
        shuffle=False,
        regression=True,
    )
    data_module.setup_data(X, y, offset=offset)

    np.testing.assert_array_equal(
        np.asarray(data_module.offset_train),
        data_module.X_train["x"].to_numpy(),
    )
    np.testing.assert_array_equal(
        np.asarray(data_module.offset_val),
        data_module.X_val["x"].to_numpy(),
    )


def test_explicit_validation_requires_offset_val():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=40)})
    y = rng.normal(size=40)

    data_module = NAMpyDataModule(
        preprocessor=Preprocessor(numerical_preprocessing="standardization"),
        batch_size=16,
        shuffle=False,
        regression=True,
    )
    with pytest.raises(ValueError, match="offset_val is required"):
        data_module.setup_data(
            X.iloc[:30], y[:30], X_val=X.iloc[30:], y_val=y[30:], offset=y[:30]
        )


def test_gaussian_offset_shifts_fitted_solution(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = 3.0 + rng.normal(scale=0.05, size=n)

    def _fit(offset):
        torch.manual_seed(0)
        estimator = LinRegRegressor(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
        estimator.set_params(training=replace(estimator.training or GradientTrainingConfig(), max_epochs=60, patience=60, lr=0.05, batch_size=64, checkpoint_path=str(tmp_path), trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0})).fit(X, y, offset=offset)
        return float(np.mean(estimator.predict(X)))

    mean_without_offset = _fit(None)
    # With the full signal supplied as offset, the network should learn ~0.
    mean_with_offset = _fit(np.full(n, 3.0))

    assert mean_without_offset > 2.0
    assert abs(mean_with_offset) < 1.0


def test_custom_loss_fct_reaches_task_module(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
    y = np.exp(0.5 * X["x"].to_numpy())

    loss = nn.PoissonNLLLoss(log_input=True)
    estimator = LinRegRegressor(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
    estimator.set_params(training=replace(estimator.training or GradientTrainingConfig(), max_epochs=1, patience=1, checkpoint_path=str(tmp_path), trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0}), objective=replace(estimator.objective, loss_fct=loss)).fit(X, y)

    assert estimator.model.objective.loss_fct is not loss
    assert type(estimator.model.objective.loss_fct) is type(loss)


def test_custom_loss_fct_rejected_outside_regression():
    from nampy.models.classifier import NeuralClassifier
    from nampy.models.linreg import LinRegClassifier

    assert issubclass(LinRegClassifier, NeuralClassifier)
    estimator = LinRegClassifier(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
    estimator.set_params(objective=replace(estimator.objective, loss_fct=nn.MSELoss()))
    with pytest.raises(ValueError, match="only supported for regression"):
        estimator._build_training_plan(np.array([0, 1, 0, 1]), None)
