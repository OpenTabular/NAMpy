"""GAMNetClassifier: joint compiled-terms + net binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from nampy.hybrid import GAMNetClassifier
from nampy.neural.configs.linreg_config import DefaultLinRegConfig
from nampy.neural.modules.linreg import LinReg

_KW = {
    "max_epochs": 25,
    "patience": 25,
    "lr": 5e-2,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def _labeled_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)})
    eta = 1.2 * np.sin(3.0 * data["x0"]) + 1.0 * data["x3"]
    labels = np.where(
        rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))) == 1, "yes", "no"
    )
    data["y"] = np.sin(3.0 * data["x0"])  # numeric stand-in for the formula
    return data, labels


def _classifier():
    return GAMNetClassifier(
        "y ~ s(x0, k=6)",
        LinReg,
        DefaultLinRegConfig,
        lam=[0.5],
        numerical_preprocessing="standardization",
    )


def test_string_labels_round_trip(tmp_path):
    data, labels = _labeled_data()
    estimator = _classifier()
    estimator.fit(
        data,
        labels,
        neural_features=["x3"],
        checkpoint_path=str(tmp_path),
        **_KW,
    )

    assert is_classifier(estimator)
    assert list(estimator.classes_) == ["no", "yes"]

    proba = estimator.predict_proba(data)
    assert proba.shape == (len(data), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    predictions = estimator.predict(data)
    assert set(predictions) <= {"no", "yes"}
    assert estimator.score(data, labels) > 0.6


def test_persistence_round_trip(tmp_path):
    data, labels = _labeled_data(n=120)
    estimator = _classifier()
    estimator.fit(
        data,
        labels,
        neural_features=["x3"],
        checkpoint_path=str(tmp_path),
        **dict(_KW, max_epochs=3),
    )
    expected = estimator.predict_proba(data)

    path = estimator.save_model(tmp_path / "gamnet_clf.nampy")
    restored = GAMNetClassifier.load_model(path)
    np.testing.assert_allclose(restored.predict_proba(data), expected, atol=1e-6)


def test_multiclass_targets_rejected(tmp_path):
    data, _ = _labeled_data(n=90)
    labels = np.array(["a", "b", "c"] * 30)
    estimator = _classifier()
    with pytest.raises(ValueError, match="binary targets only"):
        estimator.fit(
            data,
            labels,
            neural_features=["x3"],
            checkpoint_path=str(tmp_path),
            **dict(_KW, max_epochs=1),
        )
