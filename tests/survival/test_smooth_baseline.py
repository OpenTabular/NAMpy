"""Smooth piecewise baseline composition, independent mgcv endpoints and contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from nampy import SurvivalGAM
from nampy.survival import (
    SurvivalTarget,
    bootstrap_survival,
    prepare_survival_curve_interval,
    prepare_survival_effect_interval,
    prepare_survival_effects,
)


def subjects():
    i = np.arange(60)
    X = pd.DataFrame({"x": np.sin(i*.7)})
    target = SurvivalTarget(.2+(i*13 % 47)/10, i % 4 != 1, np.resize([0, .05, .1], 60))
    return X, target, np.resize([1, 2, .5, 0], 60), .04*X.x.to_numpy()


@pytest.mark.parametrize("route", ["fixed", "reml"])
@pytest.mark.parametrize("eigen_roundoff", [0.0, 1e-14, -1e-14])
def test_smooth_baseline_matches_independent_mgcv_prediction_and_smoothing(route, eigen_roundoff, monkeypatch):
    if eigen_roundoff:
        original = np.linalg.eigh

        def perturbed(matrix):
            values, vectors = original(matrix)
            rotation = np.eye(len(values))
            for i in range(len(values) - 1):
                rotation[i, i + 1] = eigen_roundoff
                rotation[i + 1, i] = -eigen_roundoff
            return values, vectors @ rotation

        monkeypatch.setattr(np.linalg, "eigh", perturbed)
    X, y, weight, offset = subjects()
    # All five unique endpoints define the basis, without a truncated eigenbasis
    # whose coordinate-dependent penalty scaling changes under roundoff.
    model = SurvivalGAM(cuts=np.arange(6), baseline="smooth", k=5, formula="~x",
        optimize_smoothing=route == "reml", smoothing_params=[.4] if route == "fixed" else None,
        irls_tol=1e-12, control={"epsilon": 1e-12, "newton": {"conv.tol": 1e-10}}).fit(X, y, sample_weight=weight, offset=offset)
    reference = pd.read_csv(Path(__file__).parents[1]/"reference_fixtures"/"survival_smooth_baseline.csv")
    reference = reference[reference.route == route]
    query = pd.DataFrame({"x": [-.5, 0, .8]})
    prediction = model.predict_distribution(query, offset=np.zeros(3))
    assert_allclose(model.predict_link(query, offset=np.zeros(3)).ravel(), reference.link, atol=2e-7, rtol=2e-7)
    assert_allclose(prediction.hazard_rates.ravel(), reference.rate, atol=2e-7, rtol=2e-7)
    assert_allclose(prediction.survival([1, 2, 3, 4, 5]).ravel(), reference.survival, atol=2e-7, rtol=2e-7)
    if route == "reml":
        assert_allclose(model.gam_.smoothing_params, [reference.sp.iloc[0]], rtol=3e-5, atol=1e-6)
    assert_allclose(model.native_poisson_log_likelihood_, reference.loglik.iloc[0], atol=2e-7)


def test_smooth_baseline_reconstruction_effects_joint_inference_bootstrap_and_persistence(tmp_path):
    X, y, weight, offset = subjects()
    model = SurvivalGAM(cuts=np.arange(6), baseline="smooth", k=4, formula="~x",
                        optimize_smoothing=False, smoothing_params=[.4]).fit(X, y, sample_weight=weight, offset=offset)
    query = X.iloc[:3]
    kwargs = {"offset": np.zeros(3)}
    components = model.predict_components(query, **kwargs).prediction
    assert_allclose(components.intercept+components.offset+sum(components.terms.values()), components.link, atol=1e-12)
    effects = prepare_survival_effects(model, query, times=[0, 1, 3], **kwargs)
    assert len(effects.effects) == 1 and next(iter(effects.features.values())) == ("x",)
    interval = prepare_survival_curve_interval(model, query, times=[0, 1, 3], n_draws=20, seed=17, **kwargs)
    assert np.isfinite(interval.lower).all()
    contrast = prepare_survival_effect_interval(model, query, times=[0, 1, 3], n_draws=20, seed=17, **kwargs)
    assert len(contrast.intervals) == 1
    original = model.predict_link(query, **kwargs)
    restored = SurvivalGAM.load_model(model.save_model(tmp_path/"smooth.nampy"))
    assert_array_equal(restored.predict_link(query, **kwargs), original)
    assert clone(model).baseline == "smooth"
    bootstrap = bootstrap_survival(model, X, y, query, resample_indices=np.stack([np.arange(60), np.arange(59, -1, -1)]),
        sample_weight=weight, offset=offset, prediction_offset=np.zeros(3), effects={"times": [0, 1, 3]})
    assert not bootstrap.failures and len(bootstrap.effect_samples) == 2


def test_zero_event_interval_is_only_supported_with_coupled_baseline_and_failed_refits_preserve_state():
    X, y, weight, offset = subjects()
    event = y.event & ~((y.time > 2) & (y.time <= 3))
    target = SurvivalTarget(y.time, event, y.entry)
    with pytest.raises(NotImplementedError, match="Zero-event"):
        SurvivalGAM(cuts=np.arange(6), formula="~x", optimize_smoothing=False).fit(X, target, sample_weight=weight)
    model = SurvivalGAM(cuts=np.arange(6), baseline="smooth", k=4, formula="~x",
                        optimize_smoothing=False, smoothing_params=[1.]).fit(X, target, sample_weight=weight)
    before = model.predict_link(X.iloc[:2])
    with pytest.raises(ValueError, match="positive-weight exposure"):
        model.fit(X, SurvivalTarget(np.minimum(y.time, 1), event, y.entry))
    assert_array_equal(model.predict_link(X.iloc[:2]), before)


def test_invalid_temporal_routes_are_explicit():
    X, y, _, _ = subjects()
    with pytest.raises(NotImplementedError, match="Parametric"):
        SurvivalGAM(likelihood="weibull", baseline="smooth").fit(X, y)
    with pytest.raises(NotImplementedError, match="time-dependent"):
        SurvivalGAM(cuts=np.arange(6), baseline="smooth", k=4, formula="~x+survival_time:x").fit(X, y)
