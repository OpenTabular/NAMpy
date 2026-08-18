"""Both backends adopt the shared nampy.api contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.api import (
    AdditivePrediction,
    FeatureSchema,
    PersistableModel,
    SupportsCapabilities,
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
    for backend in ("gam", "neural", "hybrid"):
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
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    _fit(estimator, X, y, tmp_path)

    assert isinstance(estimator.schema_, FeatureSchema)
    assert estimator.schema_.feature_names == ("x", "z")


def test_neural_regressor_predict_components_additivity(tmp_path):
    X, y = _regression_data()
    estimator = LinRegRegressor(numerical_preprocessing="standardization")
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
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    _fit(estimator, X, labels, tmp_path)

    components = estimator.predict_components(X)
    assert components.backend == "neural"
    np.testing.assert_allclose(
        components.response, 1.0 / (1.0 + np.exp(-components.link)), atol=1e-6
    )


def test_lss_predict_components_not_implemented(tmp_path):
    X, y = _regression_data()
    estimator = LinRegLSS(numerical_preprocessing="standardization")
    _fit(estimator, X, y, tmp_path, family="normal")

    with pytest.raises(NotImplementedError):
        estimator.predict_components(X)


def test_neural_capabilities_are_truthful(tmp_path):
    reg = LinRegRegressor(numerical_preprocessing="standardization")
    clf = LinRegClassifier(numerical_preprocessing="standardization")
    lss = LinRegLSS(numerical_preprocessing="standardization")

    for estimator in (reg, clf, lss):
        assert isinstance(estimator, SupportsCapabilities)
        assert isinstance(estimator, PersistableModel)
        caps = estimator.capabilities()
        assert caps.supports_standard_errors is False
        assert caps.supports_lpmatrix is False
        assert caps.supports_term_contributions is True

    assert clf.capabilities().supports_predict_proba is True
    assert reg.capabilities().supports_predict_proba is False
