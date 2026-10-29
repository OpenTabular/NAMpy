"""Piecewise models compose the existing mgcv-parity Poisson engine."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from nampy.models import GAMRegressor, SurvivalGAM
from nampy.survival import PiecewiseExponentialDesign, SurvivalTarget


def subjects():
    X = pd.DataFrame({"x": np.tile(np.linspace(-1, 1, 12), 2)})
    y = SurvivalTarget(
        np.r_[np.linspace(0.15, 0.95, 12), np.linspace(1.1, 2.9, 12)],
        np.tile([1, 0, 1], 8),
        entry=np.tile([0, 0.02, 0.05], 8),
    )
    return X, y


def test_constant_rates_entry_weights_and_likelihood_correction():
    X, y = subjects()
    weights = np.tile([1.0, 2.0, 0.0], 8)
    estimator = SurvivalGAM([0, 1, 3], formula="~1", optimize_smoothing=False).fit(
        X, y, sample_weight=weights
    )
    design = PiecewiseExponentialDesign(y, estimator.cuts_)
    expected = (weights @ design.events) / (weights @ design.exposure)
    prediction = estimator.predict_distribution(X)
    assert_allclose(
        prediction.hazard_rates,
        np.broadcast_to(expected, prediction.hazard_rates.shape),
        rtol=1e-7,
    )
    assert_allclose(
        estimator.score(X, y, weights) * weights.sum(),
        estimator.survival_log_likelihood_,
        rtol=1e-12,
    )
    rows, intervals = np.nonzero(design.exposure)
    correction = np.sum(
        weights[rows]
        * design.events[rows, intervals]
        * np.log(design.exposure[rows, intervals])
    )
    assert_allclose(
        estimator.native_poisson_log_likelihood_ - correction,
        estimator.survival_log_likelihood_,
    )


@pytest.mark.parametrize("optimize", [False, True])
def test_expanded_poisson_coefficients_covariance_and_smoothing_unchanged(optimize):
    X, y = subjects()
    weights = np.tile([1.0, 2.0, 0.5], 8)
    offset = np.linspace(-0.2, 0.2, len(y))
    estimator = SurvivalGAM(
        [0, 1, 3],
        formula="~s(x,k=4)",
        optimize_smoothing=optimize,
        smoothing_params=None if optimize else [0.3],
    ).fit(X, y, sample_weight=weights, offset=offset)
    design = PiecewiseExponentialDesign(y, estimator.cuts_)
    rows, interval = np.nonzero(design.exposure)
    expanded = X.iloc[rows].reset_index(drop=True)
    expanded["survival_interval"] = pd.Categorical(interval, categories=[0, 1])
    expanded["survival_event"] = design.events[rows, interval]
    raw = GAMRegressor(
        formula="survival_event ~ factor(survival_interval) + s(x,k=4)",
        family="poisson",
        optimize_smoothing=optimize,
        smoothing_params=None if optimize else [0.3],
    ).fit(
        expanded,
        sample_weight=weights[rows],
        offset=np.log(design.exposure[rows, interval]) + offset[rows],
    )
    a, b = estimator.gam_.fit_result(), raw.gam_.fit_result()
    assert_array_equal(a.coef_full, b.coef_full)
    assert_array_equal(a.smoothing_params, b.smoothing_params)
    assert_allclose(a.core.edf, b.core.edf, rtol=0, atol=0)
    assert a.criterion_value == b.criterion_value
    assert_allclose(estimator.gam_.loglik(), raw.gam_.loglik(), rtol=0, atol=0)
    assert_array_equal(estimator.gam_.vcov(), raw.gam_.vcov())
    components = estimator.predict_components(X, offset=offset)
    components.prediction.validate_additive_reconstruction()
    assert_allclose(
        np.exp(components.prediction.link[:, 0]),
        estimator.predict_distribution(X, offset=offset).hazard_rates.ravel(),
    )
    assert_array_equal(components.subject_index, np.repeat(np.arange(len(X)), 2))
    with pytest.raises(ValueError, match="offset"):
        estimator.predict(X, times=[1])


def test_replication_persistence_and_failed_refit(tmp_path):
    X, y = subjects()
    weights = np.tile([1, 2, 3], 8)
    estimator = SurvivalGAM([0, 1, 3], optimize_smoothing=False).fit(
        X, y, sample_weight=weights
    )
    repeated = np.repeat(np.arange(len(y)), weights)
    duplicated = SurvivalTarget(y.time[repeated], y.event[repeated], y.entry[repeated])
    other = clone(estimator).fit(X.iloc[repeated], duplicated)
    before = estimator.predict(X, times=[0, 1, 2, 3])
    assert_allclose(before, other.predict(X, times=[0, 1, 2, 3]), rtol=1e-10)
    restored = SurvivalGAM.load_model(estimator.save_model(tmp_path / "survival.nampy"))
    assert_array_equal(before, restored.predict(X, times=[0, 1, 2, 3]))
    with pytest.raises(ValueError, match="positive-weight event"):
        estimator.fit(X, SurvivalTarget(y.time, np.zeros(len(y))))
    assert_array_equal(before, estimator.predict(X, times=[0, 1, 2, 3]))
    with pytest.raises(NotImplementedError, match="piecewise"):
        SurvivalGAM([0, 1, 3], likelihood="cox").fit(X, y)
    with pytest.raises(NotImplementedError, match="Zero-event"):
        SurvivalGAM([0, 0.01, 3]).fit(X, y)
