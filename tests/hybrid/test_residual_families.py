"""GAMResidual family matrix: composition exactness and sklearn contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score

from nampy.hybrid import GAMResidualClassifier, GAMResidualRegressor
from nampy.models.linreg import LinRegClassifier, LinRegRegressor

_KW = {
    "max_epochs": 20,
    "patience": 20,
    "lr": 5e-2,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def _base_frame(n=180, seed=0):
    rng = np.random.default_rng(seed)
    return (
        pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)}),
        rng,
    )


def _regressor():
    return LinRegRegressor(numerical_preprocessing="standardization")


def _classifier():
    return LinRegClassifier(numerical_preprocessing="standardization")


def _fit(estimator, data, tmp_path):
    kwargs = dict(_KW, checkpoint_path=str(tmp_path))
    estimator.fit(data, neural_features=["x3"], neural_fit_kwargs=kwargs)
    return estimator


def test_gaussian_identity_composition(tmp_path):
    data, rng = _base_frame()
    data["y"] = (
        np.sin(3.0 * data["x0"]) + 2.0 * data["x3"]
        + rng.normal(scale=0.1, size=len(data))
    )
    estimator = _fit(
        GAMResidualRegressor("y ~ s(x0, k=6)", _regressor()), data, tmp_path
    )
    np.testing.assert_allclose(
        estimator.predict(data), estimator.predict_link(data), atol=1e-10
    )
    assert estimator.score(data, data["y"]) > 0.9


def test_poisson_log_link_composition(tmp_path):
    data, rng = _base_frame(seed=3)
    eta = 0.6 * np.sin(3.0 * data["x0"]) + 0.8 * data["x3"]
    data["cnt"] = rng.poisson(np.exp(eta))

    estimator = _fit(
        GAMResidualRegressor(
            "cnt ~ s(x0, k=6)", _regressor(), family="poisson"
        ),
        data,
        tmp_path,
    )

    predictions = estimator.predict(data)
    assert (predictions > 0).all()
    # Response is exactly exp of the composed linear predictor.
    np.testing.assert_allclose(
        predictions, np.exp(estimator.predict_link(data)), atol=1e-10
    )
    # The Poisson NLL reached the neural training stage.
    assert isinstance(estimator.neural_.model.loss_fct, nn.PoissonNLLLoss)
    # Fitting the count signal beats the GAM-alone deviance proxy.
    assert estimator.score(data, data["cnt"]) > 0.3


def test_binomial_logit_composition(tmp_path):
    data, rng = _base_frame(seed=5)
    eta = np.sin(3.0 * data["x0"]) + 1.5 * data["x3"]
    data["b"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    estimator = _fit(
        GAMResidualClassifier("b ~ s(x0, k=6)", _classifier()), data, tmp_path
    )

    proba = estimator.predict_proba(data)
    np.testing.assert_allclose(
        proba[:, 1],
        1.0 / (1.0 + np.exp(-estimator.predict_link(data))),
        atol=1e-10,
    )
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        estimator.decision_function(data), estimator.predict_link(data)
    )
    assert set(estimator.predict(data)) <= {0, 1}
    assert estimator.score(data, data["b"]) > 0.6


def test_unsupported_families_raise(tmp_path):
    data, rng = _base_frame(n=60)
    data["y"] = rng.normal(size=len(data))
    with pytest.raises(ValueError, match="supports families"):
        GAMResidualRegressor("y ~ s(x0)", _regressor(), family="gamma").fit(
            data, neural_features=["x3"]
        )
    with pytest.raises(ValueError, match="supports families"):
        GAMResidualClassifier(
            "y ~ s(x0)", _classifier(), family="poisson"
        ).fit(data, neural_features=["x3"])


def test_tags_clone_and_params():
    reg = GAMResidualRegressor("y ~ s(x0)", _regressor(), family="poisson")
    clf = GAMResidualClassifier("y ~ s(x0)", _classifier())

    assert is_regressor(reg) and not is_classifier(reg)
    assert is_classifier(clf) and not is_regressor(clf)

    cloned = clone(reg)
    assert type(cloned) is GAMResidualRegressor
    assert cloned.family == "poisson"
    assert cloned.formula == reg.formula
    # The nested neural template is cloned too, not shared.
    assert cloned.neural is not reg.neural
    assert type(cloned.neural) is type(reg.neural)

    params = reg.get_params(deep=False)
    assert params["family"] == "poisson"
    assert params["formula"] == "y ~ s(x0)"


def test_cross_val_score_runs_on_composer(tmp_path):
    data, rng = _base_frame(n=120, seed=7)
    data["y"] = (
        np.sin(3.0 * data["x0"]) + data["x3"]
        + rng.normal(scale=0.1, size=len(data))
    )

    estimator = GAMResidualRegressor("y ~ s(x0, k=5)", _regressor())
    scores = cross_val_score(
        estimator,
        data,
        data["y"],
        cv=2,
        params={
            "neural_features": ["x3"],
            "neural_fit_kwargs": dict(
                _KW, max_epochs=5, checkpoint_path=str(tmp_path)
            ),
        },
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_evaluate_and_capabilities(tmp_path):
    data, rng = _base_frame(n=100)
    data["y"] = data["x3"] + rng.normal(scale=0.1, size=len(data))
    estimator = _fit(
        GAMResidualRegressor("y ~ s(x0, k=5)", _regressor()), data, tmp_path
    )

    scores = estimator.evaluate(data, data["y"])
    assert "Mean Squared Error" in scores

    caps = estimator.capabilities()
    assert caps.supports_term_contributions is True
    assert caps.supports_predict_proba is False
    assert (
        GAMResidualClassifier("y ~ s(x0)", _classifier())
        .capabilities()
        .supports_predict_proba
        is True
    )
