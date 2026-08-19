"""Contracts of the shared neural training engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from nampy.models._base import build_callbacks
from nampy.models.linreg import LinRegClassifier, LinRegRegressor
from nampy.models.treenam import TreeNAMRegressor


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
    assert plan.taskmodel_kwargs == {"num_classes": 1, "task": "regression"}
    assert reg.n_outputs_ == 1

    clf = LinRegClassifier(numerical_preprocessing="standardization")
    y_enc, _, plan = clf._build_training_plan(np.array(["b", "a", "b", "a"]), None)
    assert plan.datamodule_regression is False
    assert plan.taskmodel_kwargs == {"num_classes": 2, "task": "classification"}
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
