"""Both backends adopt the shared backend-neutral contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.contracts import (
    AdditivePrediction,
    FeatureSchema,
)
from nampy.models.linreg import LinRegClassifier, LinRegLSS, LinRegRegressor


def _regression_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = 2.0 * X["x"].to_numpy() - X["z"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, y


def _fit(estimator, X, y, tmp_path, **fit_kwargs):
    return estimator.fit(
        X,
        y,
        max_epochs=2,
        patience=1,
        checkpoint_path=str(tmp_path),
        **fit_kwargs,
    )


def test_additive_prediction_backend_literals():
    with pytest.raises(ValueError, match="backend"):
        AdditivePrediction(
            response=np.zeros(2),
            link=np.zeros(2),
            terms={},
            intercept=0.0,
            backend="other",
        )
    for backend in ("gam", "neural"):
        result = AdditivePrediction(
            response=np.zeros(2),
            link=np.zeros(2),
            terms={},
            intercept=0.0,
            backend=backend,
        )
        assert result.backend == backend


def test_neural_estimators_record_feature_schema(tmp_path):
    X, y = _regression_data()
    estimator = LinRegRegressor(numerical_method="standardization")
    _fit(estimator, X, y, tmp_path)

    assert isinstance(estimator.schema_, FeatureSchema)
    assert estimator.schema_.feature_names == ("x", "z")


def test_neural_regressor_predict_components_additivity(tmp_path):
    X, y = _regression_data()
    estimator = LinRegRegressor(numerical_method="standardization")
    _fit(estimator, X, y, tmp_path)

    components = estimator.predict_components(X)
    assert components.backend == "neural"
    assert set(components.terms) >= {"x", "z"}

    reconstruction = np.zeros_like(components.link)
    for value in components.terms.values():
        reconstruction = reconstruction + np.asarray(value).reshape(len(X), -1).sum(
            axis=1
        )
    reconstruction = reconstruction + np.asarray(components.intercept).sum()
    np.testing.assert_allclose(reconstruction, components.link, atol=1e-4)


def test_neural_binary_classifier_predict_components(tmp_path):
    X, y = _regression_data()
    labels = (y > np.median(y)).astype(int)
    estimator = LinRegClassifier(numerical_method="standardization")
    _fit(estimator, X, labels, tmp_path)

    components = estimator.predict_components(X)
    assert components.backend == "neural"
    np.testing.assert_allclose(
        components.response, 1.0 / (1.0 + np.exp(-components.link)), atol=1e-6
    )


def test_lss_predict_components_multicolumn(tmp_path):
    X, y = _regression_data()
    estimator = LinRegLSS(
        family="normal", numerical_method="standardization"
    )
    _fit(estimator, X, y, tmp_path)

    components = estimator.predict_components(X)
    assert components.backend == "neural"
    # One column per distribution parameter, on both scales.
    assert components.link.shape == (len(X), 2)
    assert components.response.shape == (len(X), 2)
    for value in components.terms.values():
        assert np.asarray(value).shape[0] == len(X)
    # Additivity holds on the raw (link) scale.
    reconstruction = np.zeros_like(components.link)
    for value in components.terms.values():
        reconstruction = reconstruction + np.asarray(value).reshape(len(X), -1)
    reconstruction = reconstruction + np.asarray(components.intercept)
    np.testing.assert_allclose(reconstruction, components.link, atol=1e-4)


def test_neural_interfaces_are_explicit(tmp_path):
    reg = LinRegRegressor(numerical_method="standardization")
    clf = LinRegClassifier(numerical_method="standardization")
    lss = LinRegLSS(numerical_method="standardization")

    for estimator in (reg, clf, lss):
        assert hasattr(estimator, "predict_components")

    assert hasattr(clf, "predict_proba")
    assert not hasattr(reg, "predict_proba")


def test_regressor_predict_components_multioutput(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=50), "z": rng.normal(size=50)})
    Y = np.column_stack([2.0 * X["x"], -1.0 * X["z"]])

    estimator = LinRegRegressor(numerical_method="standardization")
    _fit(estimator, X, Y, tmp_path)

    components = estimator.predict_components(X)
    assert components.link.shape == (50, 2)
    for value in components.terms.values():
        assert np.asarray(value).shape == (50, 2)
