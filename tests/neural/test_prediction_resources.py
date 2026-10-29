"""Prediction batching and fit checkpoints have explicit resource ownership."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from nampy import GradientTrainingConfig, LinRegRegressor
from nampy.neural.prediction import basis_transform, predict_raw


class _Preprocessing:
    def __init__(self, limit):
        self.limit = limit
        self.seen = []

    def preprocess_tensors(self, rows):
        assert len(rows) <= self.limit
        self.seen.extend(rows.index.tolist())
        return {}, {"x": torch.tensor(rows.to_numpy(), dtype=torch.float32)}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.child = torch.nn.Identity()

    def linear_design(self, num, cat):
        return num["x"] * self.weight

    def forward(self, num_features, cat_features):
        term = self.linear_design(num_features, cat_features)
        return {
            "output": term + 1,
            "x": term,
            "intercept": self.weight.new_tensor([1.0]),
        }


@pytest.mark.parametrize("size", [None, 7])
def test_preprocessing_is_batched_and_cpu_results_preserve_rows_and_modes(size):
    X = pd.DataFrame({"x": np.arange(2053.0)}, index=np.arange(2053)[::-1])
    model = _Model().train()
    model.child.eval()
    preprocessing = _Preprocessing(1024 if size is None else size)
    result = predict_raw(model, preprocessing, X, batch_size=size)
    assert preprocessing.seen == X.index.tolist()
    torch.testing.assert_close(
        result["output"], torch.tensor(2 * X.to_numpy() + 1, dtype=torch.float32)
    )
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in result.values()
    )
    assert model.training and not model.child.training
    np.testing.assert_array_equal(
        basis_transform(model, preprocessing, X, batch_size=size), 2 * X.to_numpy()
    )


def test_prediction_restores_modes_after_preprocessing_failure():
    model = _Model().train()
    with pytest.raises(AssertionError):
        predict_raw(
            model, _Preprocessing(1), pd.DataFrame({"x": [1.0, 2.0]}), batch_size=2
        )
    assert model.training


@pytest.mark.parametrize("fail", [False, True])
def test_default_checkpoint_root_is_removed_even_on_failure(monkeypatch, fail):
    from nampy.neural.training import gradient

    roots = []

    def execute(estimator, training, checkpoint_root):
        root = Path(checkpoint_root)
        roots.append(root)
        (root / "marker.ckpt").write_text("temporary")
        estimator.best_model_path_ = str(root / "marker.ckpt")
        if fail:
            raise RuntimeError("training failed")

    monkeypatch.setattr(gradient, "_fit_gradient", execute)
    estimator = SimpleNamespace()
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            gradient.fit_gradient(estimator, GradientTrainingConfig())
    else:
        gradient.fit_gradient(estimator, GradientTrainingConfig())
        assert estimator.best_model_path_ is None
    assert roots and not roots[0].exists()


def test_explicit_checkpoint_paths_survive_fit_and_reproduce_default_weights(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = X.x.to_numpy() ** 2
    options = {
        "max_epochs": 1,
        "random_state": 31,
        "trainer_kwargs": {
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    }
    ephemeral = LinRegRegressor(training=GradientTrainingConfig(**options)).fit(X, y)
    retained = LinRegRegressor(
        training=GradientTrainingConfig(checkpoint_path=str(tmp_path), **options)
    ).fit(X, y)
    assert ephemeral.best_model_path_ is None
    assert retained.best_model_path_ and Path(retained.best_model_path_).is_file()
    np.testing.assert_array_equal(ephemeral.predict(X), retained.predict(X))
