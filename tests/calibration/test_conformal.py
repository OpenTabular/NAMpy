"""Finite-sample conformal contracts, no interpolation or silent score changes."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy.calibration import ConformalCalibration, calibrate_conformal
from nampy.distributions import DistributionPrediction, NormalLaw


def test_absolute_scores_exact_rank_and_input_ownership():
    y = np.arange(1., 10.)
    prediction = np.zeros(9)
    fitted = calibrate_conformal(y, prediction, level=.8)
    assert fitted.rank == 8 and fitted.correction == 8
    assert fitted.correction != np.quantile(y, .8)
    y[:] = 0
    assert_array_equal(fitted.scores, np.arange(1., 10.))
    interval = fitted.interval([2., 4.])
    assert_array_equal(interval.lower, [-6, -4])
    assert_array_equal(interval.upper, [10, 12])
    assert interval.uncertainty_kind == "future_observation_conformal"
    assert interval.coverage == "marginal_under_exchangeability"
    for value in (fitted.scores, interval.lower, interval.upper):
        assert not value.flags.writeable
    assert not pickle.loads(pickle.dumps(fitted)).scores.flags.writeable


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_too_few_calibration_rows_returns_unbounded_set(method):
    predictions = np.zeros(2) if method == "absolute" else np.array([[-1., 1.]]*2)
    fitted = calibrate_conformal([0., 1.], predictions, method=method, level=.95)
    assert fitted.rank == 3 and fitted.correction == np.inf
    interval = fitted.interval(predictions)
    assert np.isneginf(interval.lower).all() and np.isposinf(interval.upper).all()
    assert not interval.empty.any()


def test_cqr_signed_scores_shrink_and_can_empty_without_repair():
    fitted = calibrate_conformal(np.zeros(9), np.array([[-10., 10.]]*9), method="quantile", level=.8)
    assert_array_equal(fitted.scores, np.full(9, -10.))
    interval = fitted.interval([[-1., 1.], [-20., 20.]])
    assert_array_equal(interval.lower, [9., -10.])
    assert_array_equal(interval.upper, [-9., 10.])
    assert_array_equal(interval.empty, [True, False])


def test_cqr_matches_paper_score_and_corrected_order_statistic():
    lower = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6.])
    upper = lower+2
    y = np.array([-3, -1, 1, 4, 7, 0, 3, 9, 8.])
    fitted = calibrate_conformal(y, np.column_stack([lower, upper]), method="quantile", level=.8)
    assert_array_equal(fitted.scores, [1, 0, -1, 1, 3, 3, 1, 2, 0])
    expected = 3  # Eighth order statistic of the independently tabulated scores.
    assert fitted.correction == expected
    interval = fitted.interval([[1, 5]])
    assert_array_equal(interval.lower, [1-expected])
    assert_array_equal(interval.upper, [5+expected])


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_exhaustive_exchangeable_holdouts_have_finite_sample_coverage(method):
    # Each of five distinct responses is equally likely to be the held-out one.
    # Fixed predictors are independent of all five responses. Enumerating the
    # holdout assignments checks coverage without Monte Carlo tolerances.
    y = np.array([-3., -1., 0., 2., 4.])
    predictions = np.zeros(5) if method == "absolute" else np.array([[-1., 1.]]*5)
    covered = 0
    for held in range(5):
        take = np.arange(5) != held
        fitted = calibrate_conformal(y[take], predictions[take], method=method, level=.6)
        interval = fitted.interval(predictions[[held]])
        covered += bool(interval.lower[0] <= y[held] <= interval.upper[0])
    assert covered == 3  # ceil((4+1)*.6)/5, including signed CQR scores.


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_shared_distribution_is_equivalent_to_explicit_predictions(method):
    prediction = DistributionPrediction(NormalLaw(), {"loc": np.arange(12.), "scale": np.linspace(.5, 2, 12)})
    values = prediction.mean() if method == "absolute" else prediction.quantile([.1, .9])
    y = np.arange(12.) + np.linspace(-2, 2, 12)
    shared = calibrate_conformal(y, prediction, method=method, level=.8)
    explicit = calibrate_conformal(y, values, method=method, level=.8)
    assert_array_equal(shared.scores, explicit.scores)
    assert_allclose(shared.interval(prediction).lower, explicit.interval(values).lower)


@pytest.mark.parametrize("y,prediction", [([], []), ([1, np.nan], [1, 2]),
    ([1], [np.inf]), ([[1]], [1]), ([1, 2], [1]), ([1], [[1]]), ([1], np.array([1+2j]))])
def test_invalid_scalar_rows_are_rejected(y, prediction):
    with pytest.raises(ValueError):
        calibrate_conformal(y, prediction)


def test_unsupported_routes_and_crossing_are_explicit():
    for predictions in ([[2, 1]], [[np.nan, 1]], [1]):
        with pytest.raises(ValueError):
            calibrate_conformal([1], predictions, method="quantile")
    for level in (0, 1, np.nan, True, "0.9", [0.9], 1j):
        with pytest.raises(ValueError, match="level"):
            calibrate_conformal([1], [1], level=level)
    with pytest.raises(NotImplementedError):
        calibrate_conformal([1], [1], method="weighted")
    with pytest.raises(ValueError, match="nonnegative"):
        ConformalCalibration([-1])
    with pytest.raises(TypeError):
        calibrate_conformal([1], [1], sample_weight=[1])
