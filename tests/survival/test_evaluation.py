"""Risk sets, tie conventions, IPCW support, and censoring separation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy.survival import (
    CensoringDistribution,
    SurvivalPrediction,
    SurvivalTarget,
    brier_score,
    concordance_index,
    survival_curve,
)


def test_entry_aware_weighted_risk_sets_and_two_distinct_estimators():
    target = SurvivalTarget([1, 2, 2, 3, 4], [1, 1, 0, 1, 0], [0, 0, 1, 2, 0])
    curve = survival_curve(target, sample_weight=[1, 2, 1, 1, 0])
    # The record entering exactly at 1/2 is excluded at that exit time.
    assert_array_equal(curve.time, [1, 2, 3])
    assert_array_equal(curve.at_risk, [3, 3, 1])
    assert_array_equal(curve.events, [1, 2, 1])
    assert_array_equal(curve.censored, [0, 1, 0])
    assert_allclose(curve.survival, [2/3, 2/9, 0])
    assert_allclose(curve.cumulative_hazard, [1/3, 1, 2])
    assert_allclose(curve.survival_at([0, .5, 1, 2, 3]), [1, 1, 2/3, 2/9, 0])
    assert_allclose(curve.cumulative_hazard_at([0, 1]), [0, 1/3])
    assert not curve.survival.flags.writeable
    with pytest.raises(ValueError, match="support"):
        curve.survival_at([4])


@pytest.mark.parametrize("event", [[0, 0, 0], [1, 1, 1]])
def test_descriptive_degenerate_event_status_is_valid(event):
    curve = survival_curve(SurvivalTarget([1, 2, 3], event))
    assert_allclose(curve.survival, [1, 1, 1] if not any(event) else [2/3, 1/3, 0])


def test_frequency_replication_and_permutation_of_all_metrics():
    target = SurvivalTarget([1, 2, 2, 3, 5], [1, 1, 0, 1, 1])
    weight = np.array([2, 3, 1, 0, 2])
    repeated = np.repeat(np.arange(len(target)), weight)
    expanded = SurvivalTarget(target.time[repeated], target.event[repeated])
    curve, replicate = survival_curve(target, sample_weight=weight), survival_curve(expanded)
    assert_allclose(curve.survival, replicate.survival)
    assert_allclose(curve.cumulative_hazard, replicate.cumulative_hazard)
    censoring = CensoringDistribution.fit(target, sample_weight=weight)
    assert_allclose(censoring.probability, CensoringDistribution.fit(expanded).probability)
    risk = np.array([4, 2, 3, 1, 2])
    result = concordance_index(target, risk, higher_risk=True, sample_weight=weight)
    replicated = concordance_index(expanded, risk[repeated], higher_risk=True)
    assert_allclose(result.concordance, replicated.concordance)
    assert_allclose(result.comparable, replicated.comparable)
    prediction = SurvivalPrediction(np.full((len(target), 1), .2), [0, 6])
    expanded_prediction = SurvivalPrediction(prediction.hazard_rates[repeated], [0, 6])
    times = [1, 2, 4]
    score = brier_score(prediction, target, times, censoring=censoring, sample_weight=weight)
    assert_allclose(score.score, brier_score(expanded_prediction, expanded, times,
                                           censoring=censoring).score)
    order = np.array([4, 0, 3, 1, 2])
    assert_allclose(survival_curve(target.take(order), sample_weight=weight[order]).survival,
                    curve.survival)


def test_reverse_km_removes_tied_events_before_censoring():
    training = SurvivalTarget([1, 2, 2, 4], [1, 1, 0, 1])
    estimate = CensoringDistribution.fit(training)
    assert_array_equal(estimate.time, [1, 2, 4])
    assert_allclose(estimate.probability, [1, .5, .5])
    assert_allclose(estimate.survival_at([0, 1, 1.9, 2, 4]), [1, 1, 1, .5, .5])
    with pytest.raises(ValueError, match="support"):
        estimate.survival_at([4.1])
    zero = CensoringDistribution.fit(SurvivalTarget([1, 2], [1, 0]))
    with pytest.raises(ValueError, match="positive"):
        zero.survival_at([2])


def test_brier_hand_calculation_and_training_outcomes_are_not_refit(monkeypatch):
    censoring = CensoringDistribution.fit(SurvivalTarget([1, 2, 2, 4], [1, 1, 0, 1]))
    target = SurvivalTarget([1, 2, 3], [1, 0, 1])
    prediction = SurvivalPrediction(np.array([[.1], [.2], [.3]]), [0, 4])
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not fit censoring")
    monkeypatch.setattr(CensoringDistribution, "fit", forbidden)
    result = brier_score(prediction, target, [1, 2], censoring=censoring)
    s = prediction.survival([1, 2])
    expected = [(s[0, 0]**2+(1-s[1, 0])**2+(1-s[2, 0])**2)/3,
                (s[0, 1]**2+2*(1-s[2, 1])**2)/3]
    assert_allclose(result.score, expected)
    assert_allclose(result.integrated(), np.mean(expected))
    assert not result.score.flags.writeable
    with pytest.raises(ValueError, match="two times"):
        brier_score(prediction, target, [1], censoring=censoring).integrated()


def test_without_censoring_brier_reduces_to_binary_squared_error():
    target = SurvivalTarget([1, 2, 3, 5], [1, 1, 1, 1])
    censoring = CensoringDistribution.fit(target)
    prediction = SurvivalPrediction(np.full((4, 1), .4), [0, 6])
    times = np.array([0., 1., 2., 4.])
    s = prediction.survival(times)
    score = brier_score(prediction, target, times, censoring=censoring)
    assert_allclose(score.score, np.mean((s-(target.time[:, None] > times))**2, axis=0))


def test_concordance_ties_and_explicit_direction():
    target = SurvivalTarget([1, 1, 2, 3], [1, 0, 1, 1])
    result = concordance_index(target, [3, 3, 2, 1], higher_risk=True)
    assert result.comparable == 4 and result.tied_risk == 1 and result.tied_time == 1
    assert result.concordance == 3.5/4
    reversed_result = concordance_index(target, [3, 3, 2, 1], higher_risk=False)
    assert reversed_result.concordance == 1-result.concordance
    events_tied = SurvivalTarget([1, 1, 2], [1, 1, 1])
    assert concordance_index(events_tied, [3, 2, 1], higher_risk=True).comparable == 2
    with pytest.raises(ValueError, match="comparable"):
        concordance_index(SurvivalTarget([1, 2], [0, 0]), [2, 1], higher_risk=True)


def test_entry_support_and_zero_ipcw_are_explicit():
    entered = SurvivalTarget([1, 2], [1, 0], [.1, 0])
    prediction = SurvivalPrediction(np.full((2, 1), .2), [0, 4])
    censoring = CensoringDistribution.fit(SurvivalTarget([1, 3], [0, 1]))
    with pytest.raises(NotImplementedError, match="Delayed entry"):
        CensoringDistribution.fit(entered)
    with pytest.raises(NotImplementedError, match="Delayed entry"):
        concordance_index(entered, [2, 1], higher_risk=True)
    with pytest.raises(NotImplementedError, match="Delayed entry"):
        brier_score(prediction, entered, [1], censoring=censoring)
    target = SurvivalTarget([1, 3], [1, 0])
    with pytest.raises(TypeError, match="explicitly"):
        brier_score(prediction, target, [1], censoring=None)
    with pytest.raises(ValueError, match="precede"):
        brier_score(prediction, target, [3], censoring=censoring)
    zero = CensoringDistribution.fit(SurvivalTarget([.5, 1], [1, 0]))
    with pytest.raises(ValueError, match="positive"):
        brier_score(prediction, target, [1], censoring=zero)


def test_censoring_support_only_required_for_used_events():
    censoring = CensoringDistribution.fit(SurvivalTarget([1, 3], [1, 1]))
    # Exit at 10 is beyond training follow-up, but it is only a control at 2.
    target = SurvivalTarget([1, 10], [1, 1])
    prediction = SurvivalPrediction(np.full((2, 1), .2), [0, 12])
    assert np.isfinite(brier_score(prediction, target, [2], censoring=censoring).score).all()
    with pytest.raises(ValueError, match="support"):
        brier_score(prediction, target, [4], censoring=censoring)
