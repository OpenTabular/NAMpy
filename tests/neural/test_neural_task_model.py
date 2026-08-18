from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from nampy.neural.training.lightning_wrapper import TaskModel
from nampy.models.linreg import LinRegRegressor


class _PenalizedTwoOutputModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, num_features, cat_features):
        batch_size = next(iter(num_features.values())).shape[0]
        output = next(iter(num_features.values())).new_zeros(
            (batch_size, self.num_classes)
        )
        return {
            "output": output,
            "smoothness_penalty": output.new_tensor(9.0),
        }


def _task_config():
    return SimpleNamespace(
        lr=1e-4,
        lr_patience=2,
        weight_decay=0.0,
        lr_factor=0.5,
    )


def test_task_model_keeps_regression_width_and_reports_unregularized_rmse():
    task_model = TaskModel(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=2,
        task="regression",
    )
    logged = {}

    def capture_log(name, value, **kwargs):
        logged[name] = value.detach()

    task_model.log = capture_log
    batch = (
        {},
        {"x": torch.tensor([[0.0], [1.0], [2.0]])},
        torch.ones((3, 2)),
    )

    objective = task_model.test_step(batch, batch_idx=0)

    assert task_model.task_kind == "regression"
    assert task_model.output_dim == 2
    assert objective == pytest.approx(10.0)
    assert logged["test_loss"] == pytest.approx(10.0)
    assert logged["test_rmse"] == pytest.approx(1.0)


def test_linreg_regressor_fits_and_predicts_multiple_targets(tmp_path):
    x = np.linspace(-1.0, 1.0, 40)
    data = pd.DataFrame({"x": x, "z": np.cos(x)})
    targets = np.column_stack((2.0 * x + 0.5, -x + 0.25 * np.cos(x)))
    estimator = LinRegRegressor(numerical_preprocessing="standardization")

    fitted = estimator.fit(
        data,
        targets,
        max_epochs=1,
        batch_size=10,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    predictions = fitted.predict(data)

    assert fitted is estimator
    assert fitted.model.task_kind == "regression"
    assert fitted.model.output_dim == 2
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()
