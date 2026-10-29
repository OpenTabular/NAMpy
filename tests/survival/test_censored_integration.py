"""Uncensored references, neural task reuse and unsupported censoring routes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    NAMSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.survival import SurvivalTarget
from tests.survival.test_censored_fitting import sample


def test_centering_keeps_original_neural_precision_with_double_offsets():
    from nampy.contracts import AdditivePrediction
    from nampy.inspection.contributions import center_additive_prediction

    a = np.array([[.11, -.17], [.21, .31]], dtype=np.float32)
    b = np.array([[.35, .09], [-.2, .12]], dtype=np.float32)
    bias = np.array([.02, -.03], dtype=np.float32)
    offset = np.array([[.001, -.002], [-.004, .008]])
    link = (a+b+bias).astype(np.float64)+offset
    record = AdditivePrediction(link=link, terms={"a": a, "b": b}, intercept=bias,
                                offset=offset, backend="neural")
    centered = center_additive_prediction(record)
    assert_array_equal(centered.link, record.link)
    assert centered.intercept.dtype == np.float32
    for term in centered.terms.values():
        assert_allclose(term.mean(axis=0), 0., atol=1e-15)


def test_failed_change_of_likelihood_preserves_fitted_prediction_dispatch(tmp_path, monkeypatch):
    from nampy import LinRegSurvival

    X, target, _ = sample("weibull")
    training = GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path),
        trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})
    model = LinRegSurvival(objective=SurvivalObjectiveConfig(cuts=(0, .4, 1.2, 3)), training=training).fit(X, target)
    before = model.predict(X, times=[0, 1, 3])
    model.objective = SurvivalObjectiveConfig(likelihood="lognormal")
    def fail(*args, **kwargs):
        raise RuntimeError("injected changed-likelihood failure")
    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", fail)
    with pytest.raises(RuntimeError, match="changed-likelihood"):
        model.fit(X, target)
    assert model.likelihood_ == "piecewise_exponential"
    assert_array_equal(model.predict(X, times=[0, 1, 3]), before)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_uncensored_endpoint_retains_existing_r_positive_family_reference(name):
    path = (
        Path(__file__).parents[1]
        / "reference_fixtures"
        / "positive_lss"
        / (name + "fit.csv")
    )
    frame = pd.read_csv(path)
    X = frame[["x"]]
    target = SurvivalTarget(frame.y, np.ones(len(frame)))
    model = SurvivalGAM(likelihood=name, formula="~x", optimize_smoothing=False).fit(
        X, target
    )
    expected = frame[["first", "second"]].to_numpy()
    actual = model.predict_distribution(X).distribution
    assert_allclose(
        np.column_stack(list(actual.parameters.values())), expected, atol=2e-5, rtol=0
    )
    assert_allclose(actual.log_prob(frame.y), frame.log_density, atol=2e-5, rtol=0)
    with pytest.raises(NotImplementedError, match="residuals"):
        model.gam_.residuals(type="response")


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_named_architecture_offsets_refit_transactions_and_native_rejection(
    name, tmp_path, monkeypatch
):
    X, target, weights = sample(name)
    training = GradientTrainingConfig(
        max_epochs=2,
        batch_size=32,
        checkpoint_path=str(tmp_path),
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    objective = SurvivalObjectiveConfig(likelihood=name)
    model = NAMSurvival(
        objective=objective,
        training=training,
        architecture_config=DefaultNAMConfig(layer_sizes=[4], dropout=0),
    )
    offsets = np.column_stack(
        [np.linspace(-0.1, 0.1, len(X)), np.linspace(0.05, -0.05, len(X))]
    )
    model.fit(
        X,
        target,
        X_val=X.copy(),
        y_val=target,
        sample_weight=weights,
        offset=offsets,
        offset_val=offsets,
    )
    before = model.predict(X, times=[0, 1, 3], offset=offsets)
    parts = model.predict_components(X, offset=offsets, center=True, reference_X=X)
    # Construction validates at the stored neural precision. The optional
    # validator's explicit defaults are stricter than float32 near zero.
    assert parts.intercept.dtype == np.float32
    assert_allclose(parts.link, model.predict_link(X, offset=offsets))
    fitted = model.model

    def fail(*args, **kwargs):
        raise RuntimeError("injected censored fit failure")

    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.fit(X, target, offset=offsets)
    assert model.model is fitted
    assert_array_equal(model.predict(X, times=[0, 1, 3], offset=offsets), before)
    with pytest.raises(NotImplementedError, match="gradient"):
        NAMSurvival(objective=objective, training=FixedDesignTrainingConfig()).fit(
            X, target
        )
    with pytest.raises(ValueError, match="offset"):
        model.predict(X, times=[1])
    model.training = replace(training, warm_start=True)
    model.objective = SurvivalObjectiveConfig(
        likelihood="lognormal" if name == "weibull" else "weibull"
    )
    with pytest.raises(ValueError, match="unchanged survival likelihood"):
        model.fit(X, target, offset=offsets)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_unused_cuts_bad_offsets_and_degenerate_fitting_are_rejected(name):
    X, target, _ = sample(name)
    with pytest.raises(ValueError, match="does not use cuts"):
        SurvivalGAM(cuts=[0, 1, 3], likelihood=name, optimize_smoothing=False).fit(
            X, target
        )
    with pytest.raises(ValueError, match="does not use cuts"):
        SurvivalObjectiveConfig(cuts=(0, 1, 3), likelihood=name)
    with pytest.raises(ValueError, match="offset"):
        SurvivalGAM(likelihood=name, optimize_smoothing=False).fit(
            X, target, offset=np.zeros(len(X))
        )
    with pytest.raises(NotImplementedError, match="two distinct"):
        SurvivalGAM(likelihood=name, optimize_smoothing=False).fit(
            X, SurvivalTarget(np.ones(len(X)), np.ones(len(X)))
        )
