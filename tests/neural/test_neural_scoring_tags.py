"""Sklearn scoring and tag contracts without mixin classes.

The estimators do not inherit RegressorMixin/ClassifierMixin; ``score`` and
``__sklearn_tags__`` are hand-written. These tests pin that contract so an
sklearn upgrade that changes the tags API fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import get_tags

from nampy.models.linreg import LinRegClassifier, LinRegLSS, LinRegRegressor


def _regression_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = 2.0 * X["x"].to_numpy() - X["z"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, y


def _classification_data(n=80, seed=0, labels=("cat", "dog")):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    raw = (X["x"].to_numpy() + 0.3 * rng.normal(size=n)) > 0
    y = np.where(raw, labels[1], labels[0])
    return X, y


def test_estimator_type_tags_without_mixins():
    reg = LinRegRegressor(numerical_preprocessing="standardization")
    clf = LinRegClassifier(numerical_preprocessing="standardization")
    lss = LinRegLSS(numerical_preprocessing="standardization")

    assert get_tags(reg).estimator_type == "regressor"
    assert get_tags(clf).estimator_type == "classifier"
    assert get_tags(lss).estimator_type is None

    assert is_regressor(reg) and not is_classifier(reg)
    assert is_classifier(clf) and not is_regressor(clf)
    assert not is_classifier(lss) and not is_regressor(lss)

    # Legacy attribute kept for pre-1.6 third-party checks.
    assert reg._estimator_type == "regressor"
    assert clf._estimator_type == "classifier"


def test_regressor_score_matches_r2(tmp_path):
    X, y = _regression_data()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.fit(X, y, max_epochs=3, patience=2, checkpoint_path=str(tmp_path))

    assert estimator.score(X, y) == pytest.approx(r2_score(y, estimator.predict(X)))


def test_classifier_string_labels_round_trip(tmp_path):
    X, y = _classification_data(labels=("cat", "dog"))
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    estimator.fit(X, y, max_epochs=3, patience=2, checkpoint_path=str(tmp_path))

    assert list(estimator.classes_) == ["cat", "dog"]

    predictions = estimator.predict(X)
    assert set(predictions) <= {"cat", "dog"}

    probabilities = estimator.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    assert estimator.score(X, y) == pytest.approx(
        accuracy_score(y, predictions)
    )


def test_classifier_noncontiguous_integer_labels(tmp_path):
    X, y = _classification_data(labels=(3, 7))
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    estimator.fit(X, y.astype(int), max_epochs=3, patience=2, checkpoint_path=str(tmp_path))

    assert list(estimator.classes_) == [3, 7]
    assert set(estimator.predict(X)) <= {3, 7}


def test_classifier_rejects_single_class():
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    with pytest.raises(ValueError, match="at least 2 classes"):
        estimator._build_training_plan(np.zeros(10), None)


def test_lss_score_is_negative_mean_nll(tmp_path):
    X, y = _regression_data()
    estimator = LinRegLSS(
        family="normal", numerical_preprocessing="standardization"
    )
    estimator.fit(
        X, y, max_epochs=3, patience=2, checkpoint_path=str(tmp_path)
    )

    score = estimator.score(X, y)
    nll = estimator.evaluate(X, y)["NLL"]
    assert score == pytest.approx(-nll)


def test_cross_val_score_runs_on_neural_estimators(tmp_path):
    X, y = _regression_data(n=60)
    reg = LinRegRegressor(numerical_preprocessing="standardization")
    scores = cross_val_score(
        reg,
        X,
        y,
        cv=2,
        params={"max_epochs": 2, "patience": 1, "checkpoint_path": str(tmp_path)},
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_cross_val_score_runs_on_lss_with_constructor_family(tmp_path):
    X, y = _regression_data(n=60)
    estimator = LinRegLSS(
        family="normal", numerical_preprocessing="standardization"
    )

    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=2,
        params={"max_epochs": 2, "patience": 1, "checkpoint_path": str(tmp_path)},
    )

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
