"""Shared bootstrap bands and paired contrasts against independent R glm refits."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    NAMSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.diagnostics import prepare_survival_effects
from nampy.survival import SurvivalTarget, bootstrap_survival
from tests.survival.test_subject_bootstrap import neural_data


@pytest.fixture(scope="module")
def r_result():
    root = Path(__file__).parents[1] / "reference_fixtures"
    X = pd.DataFrame({"x": [-.8, -.2, .3, -.4, .7, 1.1]})
    y = SurvivalTarget([.3, .8, 1.4, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1], [0, .2, .1, .4, 1, .5])
    weight, offset = np.array([1, 2, .5, 1, 0, 3]), np.array([.1, -.1, .03, -.04, .2, 0])
    model = SurvivalGAM(cuts=[0, 1, 4], formula="~x", optimize_smoothing=False, irls_tol=1e-10).fit(
        X, y, sample_weight=weight, offset=offset)
    result = bootstrap_survival(model, X, y, pd.DataFrame({"x": [-.6, .2, .9]}),
        sample_weight=weight, offset=offset, prediction_offset=[-.03, .02, .05],
        resample_indices=pd.read_csv(root / "survival_bootstrap_indices.csv").to_numpy(),
        effects={"times": [0, .5, 1, 2, 4], "conditional_on": [0, .5, 1],
                 "reference_X": pd.DataFrame({"x": [-.7, .4, 1]}),
                 "reference_weight": [1, 3, 2], "reference_offset": np.zeros(3)})
    return result, pd.read_csv(root / "survival_bootstrap_bands.csv")


@pytest.mark.parametrize("kind", ["survival", "hazard", "rmst", "effect"])
def test_bootstrap_bands_match_independent_r_glm(r_result, kind):
    result, reference = r_result
    expected = reference[(reference.kind == kind) & (reference.replicate == -1)]
    if kind == "effect":
        percentile = next(iter(result.effect_interval().intervals.values()))
        band = next(iter(result.effect_interval(interval="simultaneous").intervals.values()))
        name = next(iter(result.point_effects.effects))
        for i, prepared in enumerate(result.effect_samples):
            assert_allclose(prepared.effects[name].ravel(),
                reference[(reference.kind == kind) & (reference.replicate == i)].value,
                atol=2e-8, rtol=2e-8, equal_nan=True)
    else:
        kwargs = {"grid": [0, .5, 1, 2, 4], "conditional_on": [0, .5, 1], "estimand": kind}
        percentile = result.percentile_interval(**kwargs)
        band = result.simultaneous_interval(**kwargs)
    for actual, column in [(band.estimate, "value"), (percentile.lower, "lower"),
                           (percentile.upper, "upper"), (band.lower, "sim_lower"),
                           (band.upper, "sim_upper"), (band.standard_error, "se")]:
        assert_allclose(actual.ravel(), expected[column], atol=2e-8, rtol=2e-8, equal_nan=True)
    assert_allclose(band.critical_value, expected.critical.iloc[0], atol=2e-8, rtol=2e-8)
    assert band.coverage == "simultaneous_bootstrap"
    assert band.uncertainty_kind == "subject_bootstrap"


@pytest.mark.parametrize("estimator_class", [LinRegSurvival, NAMSurvival])
@pytest.mark.parametrize("name", ["piecewise_exponential", "weibull", "lognormal"])
def test_neural_bootstrap_bands_and_paired_effects(name, estimator_class, tmp_path):
    X, y, X_val, y_val = neural_data(name)
    extra = {}
    if estimator_class is NAMSurvival:
        from nampy.neural.configs.nam_config import DefaultNAMConfig
        extra["architecture_config"] = DefaultNAMConfig(layer_sizes=[4], dropout=0)
    training = GradientTrainingConfig(max_epochs=1, batch_size=128, random_state=17,
        checkpoint_path=str(tmp_path), trainer_kwargs={"logger": False, "enable_progress_bar": False,
                                                     "enable_model_summary": False})
    model = estimator_class(objective=SurvivalObjectiveConfig(likelihood=name,
        cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None), training=training, **extra).fit(
            X, y, X_val=X_val, y_val=y_val)
    effects = {"times": [0, .5, 1, 2], "conditional_on": .2,
               "reference_X": X.iloc[4:7], "reference_weight": [1, 2, 3]}
    selected = np.arange(len(X))
    selected[-1] = selected[-2]
    result = bootstrap_survival(model, X, y, X.iloc[:4], X_val=X_val, y_val=y_val,
        resample_indices=np.stack([np.arange(len(X)), selected]), effects=effects,
        training_randomness="independent" if name == "lognormal" else "fixed", seed=42)
    point = prepare_survival_effects(model, X.iloc[:4], **effects)
    for term in point.effects:
        assert_array_equal(result.point_effects.effects[term], point.effects[term])
        if name != "lognormal":
            assert_allclose(result.effect_samples[0].effects[term], point.effects[term], atol=2e-7, equal_nan=True)
    for kind in ("survival", "hazard", "rmst"):
        band = result.simultaneous_interval(grid=[.5, 1, 2], conditional_on=.2, estimand=kind)
        assert np.isfinite(band.lower).all() and np.isfinite(band.upper).all()
        assert np.all(band.lower <= band.estimate + 4*np.finfo(float).eps)
        assert np.all(band.estimate <= band.upper + 4*np.finfo(float).eps)
        assert band.uncertainty_kind == result.uncertainty_kind
    bands = result.effect_interval(interval="simultaneous")
    for band in bands.intervals.values():
        assert band.scope == "all_selected_terms_subjects_and_times"
        assert band.coverage == "simultaneous_bootstrap"
        assert band.uncertainty_kind == result.uncertainty_kind
    assert not list(tmp_path.rglob("survival-bootstrap-*"))


def test_bootstrap_effect_capture_failures_and_unsupported_inference(r_result, monkeypatch):
    result, _ = r_result
    with pytest.raises(ValueError, match="Capture effects"):
        replace(result, point_effects=None, effect_samples=()).effect_interval()
    with pytest.raises(NotImplementedError, match="survival, hazard or RMST"):
        result.simultaneous_interval(grid=[1], estimand="density")
    with pytest.raises(ValueError, match="valid"):
        result.simultaneous_interval(grid=[0], conditional_on=1)
    import nampy.survival.bootstrap as implementation
    original = implementation.prepare_survival_effects
    calls = 0
    def incompatible(*args, **kwargs):
        nonlocal calls
        calls += 1
        prepared = original(*args, **kwargs)
        return prepared if calls == 1 else replace(prepared, predictor_names=("changed",))
    monkeypatch.setattr(implementation, "prepare_survival_effects", incompatible)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 20)})
    y = SurvivalTarget(np.linspace(.1, 2, 20), np.ones(20))
    model = SurvivalGAM(cuts=[0, 1, 3], formula="~x", optimize_smoothing=False).fit(X, y)
    failed = bootstrap_survival(model, X, y, X.iloc[:2],
        resample_indices=np.stack([np.arange(20)]*2), effects={"times": [0, 1, 2]}, on_failure="record")
    assert not failed.predictions and not failed.effect_samples
    assert all(f.stage == "effect_preparation" and f.error_type == "NotImplementedError" for f in failed.failures)
    for operation in (failed.effect_interval, lambda: failed.simultaneous_interval(grid=[1]),
                      lambda: failed.percentile_interval(grid=[1])):
        with pytest.raises(ValueError, match="every requested replicate"):
            operation()


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_censored_refit_contrasts_share_both_predictors_and_fixed_reference(name):
    from tests.survival.test_censored_fitting import sample
    X, y, weight = sample(name)
    names = ("scale", "shape") if name == "weibull" else ("loc", "scale")
    options = {"likelihood": name, "formula": dict.fromkeys(names, "~x"), "optimize_smoothing": False}
    model = SurvivalGAM(**options).fit(X, y, sample_weight=weight)
    selected = np.arange(len(X))
    selected[-1] = selected[-2]
    indices = np.stack([np.arange(len(X)), selected])
    effects = {"times": [0, .5, 1, 2], "reference_X": X.iloc[5:9],
               "reference_weight": [1, 0, 3, 2], "conditional_on": [0, .2, .5]}
    result = bootstrap_survival(model, X, y, X.iloc[:3], sample_weight=weight,
                                resample_indices=indices, effects=effects)
    assert len(result.point_effects.predictor_names) == 2
    for b, rows in enumerate(indices):
        direct = SurvivalGAM(**options).fit(X.iloc[rows].reset_index(drop=True),
            SurvivalTarget(y.time[rows], y.event[rows], y.entry[rows]), sample_weight=weight[rows])
        paired = prepare_survival_effects(direct, X.iloc[:3], **effects)
        for term in paired.effects:
            assert_array_equal(result.effect_samples[b].effects[term], paired.effects[term])
    joint = result.effect_interval(interval="simultaneous")
    critical = {band.critical_value for band in joint.intervals.values()}
    assert len(critical) == 1
    for band in joint.intervals.values():
        assert band.scope == "all_selected_terms_subjects_and_times"


def test_zero_variance_and_mismatched_constant_bootstrap_are_explicit(r_result):
    result, _ = r_result
    identity = replace(result, predictions=(result.point_prediction,)*2,
                        effect_samples=(result.point_effects,)*2)
    band = identity.simultaneous_interval(grid=[0, 1, 2])
    assert band.critical_value == 0
    assert_allclose(band.lower, band.estimate, atol=2e-16, rtol=0)
    for value in identity.effect_interval(interval="simultaneous").intervals.values():
        assert value.critical_value == 0
        assert_array_equal(value.lower, value.estimate)
    mismatched = replace(result, predictions=(result.predictions[0],)*2)
    with pytest.raises(ValueError, match="variance"):
        mismatched.simultaneous_interval(grid=[1, 2])



def test_multiple_selected_effects_share_one_bootstrap_calibration():
    from tests.survival.test_censored_fitting import sample
    X, y, weight = sample("weibull")
    X["z"] = np.sin(np.arange(len(X)))
    model = SurvivalGAM(cuts=[0, 1, 3], formula="~x+z", optimize_smoothing=False).fit(X, y, sample_weight=weight)
    selected = np.arange(len(X))
    selected[-1] = selected[-2]
    result = bootstrap_survival(model, X, y, X.iloc[:4], sample_weight=weight,
        resample_indices=np.stack([np.arange(len(X)), selected]),
        effects={"times": [.5, 1, 2], "reference_X": X.iloc[5:9]})
    prepared = result.point_effects
    assert len(prepared.effects) == 2
    point = np.stack(list(prepared.effects.values()))
    draws = np.stack([np.stack(list(value.effects.values())) for value in result.effect_samples])
    sd = np.std(draws, axis=0, ddof=1)
    scores = np.divide(np.abs(draws-point), sd, out=np.zeros_like(draws), where=sd > 0)
    expected_critical = np.quantile(np.max(scores, axis=(1, 2, 3)), .95, method="median_unbiased")
    for i, interval in enumerate(result.effect_interval(interval="simultaneous").intervals.values()):
        assert_allclose(interval.critical_value, expected_critical)
        assert_allclose(interval.lower, np.clip(point[i]-expected_critical*sd[i], -1, 1))
