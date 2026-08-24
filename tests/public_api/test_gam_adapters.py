"""Contracts of the sklearn-style GAM adapters.

The adapters must add zero numerics on top of the raw ``nampy.gam.GAM``;
the guard test pins adapter predictions to a directly-constructed GAM with
identical hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from nampy.contracts import (
    AdditivePrediction,
)
from nampy.gam import GAM
from nampy.models import GAMClassifier, GAMRegressor


def _regression_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x0": rng.uniform(size=n), "x1": rng.uniform(size=n)})
    y = (
        np.sin(3.0 * X["x0"].to_numpy())
        + 0.5 * X["x1"].to_numpy()
        + rng.normal(scale=0.1, size=n)
    )
    return X, y


def _fixed_regressor(**overrides):
    params = {
        "k": 6,
        "optimize_smoothing": False,
        "smoothing_method": "fixed",
        "smoothing_params": [1.0, 1.0],
    }
    params.update(overrides)
    return GAMRegressor(**params)


def test_adapter_defaults_are_auto_reml():
    estimator = GAMRegressor()
    assert estimator.optimize_smoothing is True
    assert estimator.smoothing_method == "reml"


def test_adapter_matches_raw_gam_exactly():
    X, y = _regression_frame()

    adapter = _fixed_regressor().fit(X, y)
    raw = GAM(
        k=6,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0],
        basis="tp",
        fit_intercept=True,
        covariance="bayes",
        score_gamma=1.0,
        max_irls_iter=200,
        irls_tol=1e-7,
        sp_log_bounds=(-80.0, 20.0),
    )
    raw.fit(X, y)

    np.testing.assert_array_equal(adapter.predict(X), raw.predict(X))
    np.testing.assert_array_equal(
        adapter.gam_.fit_result().coef_full, raw.fit_result().coef_full
    )


def test_clone_and_tags_contract():
    reg = _fixed_regressor()
    clf = GAMClassifier(k=5)

    cloned = clone(reg)
    assert type(cloned) is GAMRegressor
    assert cloned.get_params() == reg.get_params()

    assert is_regressor(reg) and not is_classifier(reg)
    assert is_classifier(clf) and not is_regressor(clf)


def test_regressor_score_components_and_errors():
    X, y = _regression_frame()
    estimator = _fixed_regressor().fit(X, y)

    assert estimator.score(X, y) == pytest.approx(
        r2_score(y, estimator.predict(X))
    )

    components = estimator.predict_components(X)
    assert isinstance(components, AdditivePrediction)
    assert components.backend == "gam"
    assert set(components.terms) == {"x0", "x1"}
    reconstruction = sum(components.terms.values()) + components.intercept
    np.testing.assert_allclose(reconstruction, components.link, atol=1e-10)

    se = estimator.standard_errors(X)
    assert np.asarray(se).shape == (len(X),)
    lp = estimator.lpmatrix(X)
    assert lp.shape[0] == len(X)


def test_adapter_prediction_row_and_memory_options_are_forwarded():
    X, y = _regression_frame(n=70, seed=9)
    estimator = _fixed_regressor().fit(X, y)
    newdata = X.iloc[:9].copy()
    newdata.loc[3, "x0"] = np.nan

    prediction = estimator.predict(newdata, block_size=2, na_action="pass")
    assert prediction.shape == (9,)
    assert np.isnan(prediction[3])
    assert estimator.predict(newdata, na_action="omit").shape == (8,)

    components = estimator.predict_components(
        newdata, block_size=2, na_action="pass"
    )
    components.validate_additive_reconstruction()
    assert np.isnan(components.link[3])
    assert estimator.standard_errors(newdata, block_size=2).shape == (9,)
    assert estimator.lpmatrix(newdata, block_size=1).shape[0] == 9


def test_schema_validation_rejects_renamed_columns():
    X, y = _regression_frame()
    estimator = _fixed_regressor().fit(X, y)

    renamed = X.rename(columns={"x0": "different"})
    with pytest.raises(ValueError, match="name mismatch"):
        estimator.predict(renamed)

    with pytest.raises(ValueError, match="count mismatch"):
        estimator.predict(X.to_numpy()[:, :1])


def test_classifier_labels_proba_and_errors():
    X, y = _regression_frame()
    labels = np.where(y > np.median(y), "yes", "no")

    estimator = GAMClassifier(
        k=5,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0],
    ).fit(X, labels)

    assert list(estimator.classes_) == ["no", "yes"]
    probabilities = estimator.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert set(estimator.predict(X)) <= {"no", "yes"}
    assert estimator.score(X, labels) > 0.8
    assert estimator.decision_function(X).shape == (len(X),)

    with pytest.raises(ValueError, match="binary targets only"):
        GAMClassifier().fit(X, np.array([0, 1, 2] * (len(X) // 3)))


def test_estimator_interfaces_and_protocols():
    reg = _fixed_regressor()
    clf = GAMClassifier()

    assert hasattr(reg, "predict_components")
    assert hasattr(reg, "lpmatrix")
    assert hasattr(clf, "predict_proba")


def test_persistence_round_trip(tmp_path):
    X, y = _regression_frame()
    estimator = _fixed_regressor().fit(X, y)
    expected = estimator.predict(X)

    path = estimator.save_model(tmp_path / "gam.nampy")
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), expected)

    with pytest.raises(TypeError, match="not GAMClassifier"):
        GAMClassifier.load_model(path)


def test_cross_val_score_runs_on_gam_regressor():
    X, y = _regression_frame(n=90)
    scores = cross_val_score(_fixed_regressor(), X, y, cv=2)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_adapter_matches_raw_gam_for_glm_families(family):
    rng = np.random.default_rng(3)
    n = 120
    X = pd.DataFrame({"x0": rng.uniform(size=n), "x1": rng.uniform(size=n)})
    eta = 0.5 * np.sin(3.0 * X["x0"].to_numpy()) + 0.4 * X["x1"].to_numpy()
    mu = np.exp(eta)
    if family == "poisson":
        y = rng.poisson(mu).astype(float)
    else:
        y = rng.gamma(shape=4.0, scale=mu / 4.0)

    params = {
        "family": family,
        "k": 6,
        "optimize_smoothing": False,
        "smoothing_method": "fixed",
        "smoothing_params": [1.0, 1.0],
    }
    adapter = GAMRegressor(**params).fit(X, y)
    raw = GAM(
        basis="tp",
        fit_intercept=True,
        covariance="bayes",
        score_gamma=1.0,
        max_irls_iter=200,
        irls_tol=1e-7,
        sp_log_bounds=(-80.0, 20.0),
        **{key: value for key, value in params.items() if key != "family"},
        family=family,
    )
    raw.fit(X, y)

    np.testing.assert_array_equal(adapter.predict(X), raw.predict(X))
    np.testing.assert_array_equal(
        adapter.gam_.fit_result().coef_full, raw.fit_result().coef_full
    )
    # Components respect the family link on the response scale.
    components = adapter.predict_components(X)
    np.testing.assert_allclose(
        components.response,
        adapter.gam_.family.inverse_link(components.link),
        atol=1e-10,
    )
