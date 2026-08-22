"""Per-sample offset plumbing through dataset, datamodule, and TaskModule."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pretab.preprocessor import Preprocessor

from nampy.models.linreg import LinRegRegressor
from nampy.neural.data.datamodule import NAMpyDataModule
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
    task_model = TaskModule(
        model_class=_ZeroOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=1,
        task="regression",
    )
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


def test_lss_rejects_nonzero_offsets():
    family = SimpleNamespace(
        param_count=2,
        compute_loss=lambda preds, y, reduction="none": preds.sum() * 0.0,
    )
    task_model = TaskModule(
        model_class=_ZeroOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=2,
        lss=True,
        family=family,
    )
    task_model.log = lambda *args, **kwargs: None
    features = {"x": torch.tensor([[0.0], [1.0]])}
    labels = torch.ones((2, 1))

    # Zero offsets pass through.
    task_model.test_step(({}, features, labels, torch.zeros((2, 1))), 0)

    with pytest.raises(RuntimeError, match="not supported for distributional"):
        task_model.test_step(({}, features, labels, torch.ones((2, 1))), 0)


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
        estimator = LinRegRegressor(numerical_preprocessing="standardization")
        estimator.fit(
            X,
            y,
            offset=offset,
            max_epochs=60,
            patience=60,
            lr=5e-2,
            batch_size=64,
            checkpoint_path=str(tmp_path),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
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
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.fit(
        X,
        y,
        loss_fct=loss,
        max_epochs=1,
        patience=1,
        checkpoint_path=str(tmp_path),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    assert estimator.model.objective.loss_fct is loss


def test_custom_loss_fct_rejected_outside_regression():
    from nampy.models.classifier import NeuralClassifier
    from nampy.models.linreg import LinRegClassifier

    assert issubclass(LinRegClassifier, NeuralClassifier)
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    estimator._fit_loss_fct = nn.MSELoss()
    with pytest.raises(ValueError, match="only supported for regression"):
        estimator._build_training_plan(np.array([0, 1, 0, 1]), None)
