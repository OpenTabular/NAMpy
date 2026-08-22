"""Contracts of the shared neural training engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from nampy.models._base import (
    RecentStateAveraging,
    StepEarlyStopping,
    average_state_dicts,
    build_callbacks,
)
from nampy.models.linreg import LinRegClassifier, LinRegRegressor
from nampy.models.nam import NAMLSS
from nampy.models.nodegam import NodeGAMRegressor
from nampy.models.treenam import TreeNAMRegressor
from nampy.neural.objectives import (
    RegressionObjective,
    classification_objective,
)


def _make_regression_frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = 2.0 * X["x"].to_numpy() - X["z"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, y


def test_build_callbacks_honors_user_monitor_and_mode(tmp_path):
    early_stop, checkpoint = build_callbacks(
        "LinRegRegressor",
        monitor="val_custom",
        mode="max",
        patience=3,
        checkpoint_path=tmp_path,
    )

    assert early_stop.monitor == "val_custom"
    assert early_stop.mode == "max"
    assert checkpoint.monitor == "val_custom"
    assert checkpoint.mode == "max"


def test_build_callbacks_uses_unique_directory_per_fit(tmp_path):
    name = "LinRegRegressor"
    _, first = build_callbacks(
        name, monitor="val_loss", mode="min", patience=3, checkpoint_path=tmp_path
    )
    _, second = build_callbacks(
        name, monitor="val_loss", mode="min", patience=3, checkpoint_path=tmp_path
    )

    assert first.dirpath != second.dirpath
    assert str(tmp_path) in str(first.dirpath)


def test_unfitted_predict_raises_not_fitted_error():
    X, _ = _make_regression_frame()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")

    with pytest.raises(NotFittedError):
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.predict_components(X)


def test_predict_restores_module_training_flag(tmp_path):
    X, y = _make_regression_frame()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.fit(X, y, max_epochs=2, patience=1, checkpoint_path=str(tmp_path))

    estimator.model.train()
    estimator.predict(X)
    assert estimator.model.training

    estimator.model.eval()
    estimator.predict(X)
    assert not estimator.model.training


def test_training_plans_carry_task_wiring():
    reg = LinRegRegressor(numerical_preprocessing="standardization")
    y, y_val, plan = reg._build_training_plan(np.array([1.0, 2.0, 3.0]), None)
    assert plan.datamodule_regression is True
    assert isinstance(plan.objective, RegressionObjective)
    assert plan.objective.output_dim == 1
    assert reg.n_outputs_ == 1

    clf = LinRegClassifier(numerical_preprocessing="standardization")
    y_enc, _, plan = clf._build_training_plan(np.array(["b", "a", "b", "a"]), None)
    assert plan.datamodule_regression is False
    assert type(plan.objective) is type(classification_objective(2))
    assert list(clf.classes_) == ["a", "b"]
    assert set(y_enc) == {0, 1}
    assert plan.stratify is not None


def test_fit_uses_constructor_optimizer_configuration(monkeypatch):
    X, y = _make_regression_frame()
    estimator = TreeNAMRegressor(
        numerical_preprocessing="standardization",
        lr=0.02,
        lr_patience=7,
        lr_factor=0.3,
        weight_decay=0.004,
    )
    captured = []

    def fake_run_training(estimator, X, y, plan, **kwargs):
        del estimator, X, y, plan
        captured.append(kwargs)

    monkeypatch.setattr(
        "nampy.models._base.NeuralEstimatorBase._run_training", fake_run_training
    )

    estimator.fit(X, y)
    estimator.set_params(
        lr=0.03,
        lr_patience=8,
        lr_factor=0.35,
        weight_decay=0.005,
    )
    estimator.fit(X, y)
    estimator.fit(
        X,
        y,
        lr=0.05,
        lr_patience=9,
        lr_factor=0.4,
        weight_decay=0.006,
    )

    assert captured[0]["lr"] == pytest.approx(0.02)
    assert captured[0]["lr_patience"] == 7
    assert captured[0]["lr_factor"] == pytest.approx(0.3)
    assert captured[0]["weight_decay"] == pytest.approx(0.004)
    assert captured[1]["lr"] == pytest.approx(0.03)
    assert captured[1]["lr_patience"] == 8
    assert captured[1]["lr_factor"] == pytest.approx(0.35)
    assert captured[1]["weight_decay"] == pytest.approx(0.005)
    assert captured[2]["lr"] == pytest.approx(0.05)
    assert captured[2]["lr_patience"] == 9
    assert captured[2]["lr_factor"] == pytest.approx(0.4)
    assert captured[2]["weight_decay"] == pytest.approx(0.006)


def test_successive_fits_do_not_clobber_checkpoints(tmp_path):
    X, y = _make_regression_frame()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.fit(X, y, max_epochs=2, patience=1, checkpoint_path=str(tmp_path))
    estimator.fit(X, y, max_epochs=2, patience=1, checkpoint_path=str(tmp_path))

    checkpoints = list(tmp_path.rglob("*.ckpt"))
    assert len(checkpoints) == 2


def test_average_state_dicts_averages_float_and_keeps_latest_discrete_state():
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "step": torch.tensor(2)},
        {"weight": torch.tensor([3.0, 5.0]), "step": torch.tensor(4)},
    ]
    averaged = average_state_dicts(states)
    torch.testing.assert_close(averaged["weight"], torch.tensor([2.0, 4.0]))
    assert averaged["step"].item() == 4


def test_recent_state_callback_keeps_only_latest_epoch_states():
    callback = RecentStateAveraging(2)
    module = torch.nn.Linear(1, 1, bias=False)
    for value in (1.0, 3.0, 5.0):
        module.weight.data.fill_(value)
        callback.on_train_epoch_end(None, module)

    assert len(callback.states) == 2
    torch.testing.assert_close(
        callback.averaged_state_dict()["weight"], torch.tensor([[4.0]])
    )


def test_fit_forwards_generic_step_and_checkpoint_controls(monkeypatch):
    X, y = _make_regression_frame()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    captured = {}

    def fake_run_training(estimator, X, y, plan, **kwargs):
        del estimator, X, y, plan
        captured.update(kwargs)

    monkeypatch.setattr(
        "nampy.models._base.NeuralEstimatorBase._run_training", fake_run_training
    )
    estimator.fit(
        X,
        y,
        max_steps=17,
        max_time="00:01:00",
        early_stopping_steps=12,
        lr_warmup_steps=3,
        lr_decay_steps=10,
        lr_decay_factor=0.4,
        optimizer="adamw",
        optimizer_kwargs={"amsgrad": True},
        average_checkpoints=True,
        n_last_checkpoints=3,
    )

    assert captured["max_steps"] == 17
    assert captured["max_time"] == "00:01:00"
    assert captured["early_stopping_steps"] == 12
    assert captured["lr_warmup_steps"] == 3
    assert captured["lr_decay_steps"] == 10
    assert captured["lr_decay_factor"] == pytest.approx(0.4)
    assert captured["optimizer"] == "adamw"
    assert captured["optimizer_kwargs"] == {"amsgrad": True}
    assert captured["average_checkpoints"] is True
    assert captured["n_last_checkpoints"] == 3


def test_fit_combines_class_and_sample_weights_and_forwards_balancing(monkeypatch):
    X = pd.DataFrame({"x": np.arange(6.0)})
    y = np.array(["common", "common", "common", "common", "rare", "rare"])
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    captured = {}

    def fake_run_training(estimator, X, y, plan, **kwargs):
        del estimator, X, y, plan
        captured.update(kwargs)

    monkeypatch.setattr(
        "nampy.models._base.NeuralEstimatorBase._run_training", fake_run_training
    )
    estimator.fit(
        X,
        y,
        sample_weight=np.full(len(y), 2.0),
        class_weight="balanced",
        sampling_strategy="balanced",
    )

    combined = np.asarray(captured["sample_weight"])
    assert combined[y == "rare"][0] > combined[y == "common"][0]
    assert captured["sampling_strategy"] == "balanced"


def test_lss_fit_forwards_weights_to_distribution_objective(monkeypatch):
    X, y = _make_regression_frame(n=12)
    estimator = NAMLSS(numerical_preprocessing="standardization")
    captured = {}

    def fake_run_training(estimator, X, y, plan, **kwargs):
        del estimator, X, y
        captured["plan"] = plan
        captured.update(kwargs)

    monkeypatch.setattr(
        "nampy.models._base.NeuralEstimatorBase._run_training", fake_run_training
    )
    weights = np.linspace(1.0, 2.0, len(y))
    estimator.fit(X, y, sample_weight=weights)

    assert captured["plan"].objective.kind == "distributional"
    np.testing.assert_array_equal(captured["sample_weight"], weights)


def test_step_early_stopping_uses_optimizer_step_distance():
    callback = StepEarlyStopping(
        monitor="val_loss", mode="min", patience_steps=5
    )

    class Trainer:
        sanity_checking = False
        should_stop = False
        callback_metrics = {"val_loss": torch.tensor(2.0)}
        global_step = 3

    trainer = Trainer()
    callback.on_validation_end(trainer, None)
    assert callback.best_step == 3
    trainer.global_step = 7
    trainer.callback_metrics = {"val_loss": torch.tensor(2.5)}
    callback.on_validation_end(trainer, None)
    assert trainer.should_stop is False
    trainer.global_step = 8
    callback.on_validation_end(trainer, None)
    assert trainer.should_stop is True


def test_warm_start_forwards_current_model_state(monkeypatch):
    X, y = _make_regression_frame()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.model = torch.nn.Linear(2, 1)
    expected = {
        key: value.detach().clone() for key, value in estimator.model.state_dict().items()
    }
    captured = {}

    def fake_run_training(estimator, X, y, plan, **kwargs):
        del estimator, X, y, plan
        captured.update(kwargs)

    monkeypatch.setattr(
        "nampy.models._base.NeuralEstimatorBase._run_training", fake_run_training
    )
    estimator.fit(X, y, warm_start=True)

    for key, value in expected.items():
        torch.testing.assert_close(captured["warm_start_state"][key], value)


def test_nodegam_optional_masked_pretraining_transfers_compatible_state(tmp_path):
    X, y = _make_regression_frame(n=48)
    estimator = NodeGAMRegressor(
        numerical_preprocessing="standardization",
        num_trees=2,
        num_layers=1,
        depth=2,
        interaction_degree=1,
        colsample_bytree=1.0,
        output_dropout=0.0,
        last_dropout=0.0,
    )
    estimator.fit(
        X,
        y,
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=16,
        patience=1,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    assert estimator.pretrained_keys_
    assert any("feature_selection_logits" in key for key in estimator.pretrained_keys_)
    assert estimator.model.model.model[0].ga2m == 0
    explanations = estimator.explain_terms(X, max_bins=8)
    assert {"term", "contribution", "count", "importance"}.issubset(
        explanations.columns
    )
    assert not estimator.term_importance(X).empty
    assert estimator.interaction_importance(X).empty
