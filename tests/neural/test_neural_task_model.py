from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from nampy.models.linreg import LinRegRegressor
from nampy.neural.distributions.distributions import NormalDistribution
from nampy.neural.objectives import DistributionObjective
from nampy.neural.task import TaskModule


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
    task_model = TaskModule(
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
        torch.zeros((3, 1)),
    )

    objective = task_model.test_step(batch, batch_idx=0)

    assert task_model.task_kind == "regression"
    assert task_model.output_dim == 2
    assert objective == pytest.approx(10.0)
    assert logged["test_loss"] == pytest.approx(10.0)
    assert logged["test_rmse"] == pytest.approx(1.0)


def test_task_model_applies_per_sample_weights_before_penalties():
    task_model = TaskModule(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=1,
        task="regression",
    )
    task_model.log = lambda *args, **kwargs: None
    batch = (
        {},
        {"x": torch.tensor([[0.0], [1.0]])},
        torch.tensor([[1.0], [3.0]]),
        torch.zeros((2, 1)),
        torch.tensor([[1.0], [3.0]]),
    )
    objective = task_model.training_step(batch, batch_idx=0)
    # Weighted MSE is (1 * 1^2 + 3 * 3^2) / 4 = 7; penalty is 9.
    assert objective == pytest.approx(16.0)


def test_distribution_objective_applies_per_sample_weights():
    family = NormalDistribution()
    task_model = TaskModule(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        objective=DistributionObjective(family),
    )
    predictions = torch.zeros((2, 2))
    targets = torch.tensor([[0.0], [2.0]])
    weights = torch.tensor([[1.0], [3.0]])

    values = family.compute_loss(predictions, targets[:, 0], reduction="none")
    expected = (values[0] + 3.0 * values[1]) / 4.0
    actual = task_model.compute_loss(predictions, targets, sample_weight=weights)

    torch.testing.assert_close(actual, expected)


def test_task_model_supports_generic_optimizer_and_step_scheduler():
    task_model = TaskModule(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=2,
        task="regression",
        optimizer="adamw",
        optimizer_kwargs={"amsgrad": True},
        lr_warmup_steps=4,
        lr_decay_steps=10,
        lr_decay_factor=0.5,
    )
    # The tiny test model has no parameters, so add one solely to exercise
    # optimizer construction without changing the model contract above.
    task_model.test_parameter = nn.Parameter(torch.ones(()))
    configured = task_model.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.AdamW)
    assert configured["optimizer"].defaults["amsgrad"] is True
    assert configured["lr_scheduler"]["interval"] == "step"
    assert isinstance(
        configured["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LambdaLR
    )


def test_task_model_supports_upstream_inverse_sqrt_epoch_schedule():
    task_model = TaskModule(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=1,
        task="regression",
        lr_schedule="inverse_sqrt",
    )
    task_model.test_parameter = nn.Parameter(torch.ones(()))
    configured = task_model.configure_optimizers()
    scheduler = configured["lr_scheduler"]
    assert scheduler["interval"] == "epoch"
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.LambdaLR)


def test_task_model_supports_warmup_cosine_step_schedule():
    task_model = TaskModule(
        model_class=_PenalizedTwoOutputModel,
        config=_task_config(),
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=1,
        task="regression",
        lr_schedule="warmup_cosine",
        lr_warmup_steps=2,
        lr_decay_steps=6,
    )
    task_model.test_parameter = nn.Parameter(torch.ones(()))
    configured = task_model.configure_optimizers()
    scheduler = configured["lr_scheduler"]

    assert scheduler["interval"] == "step"
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    multipliers = [scheduler["scheduler"].lr_lambdas[0](step) for step in range(7)]
    assert multipliers[:2] == pytest.approx([0.5, 1.0])
    assert multipliers[2] == pytest.approx(1.0)
    assert multipliers[-1] == pytest.approx(0.0)


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
