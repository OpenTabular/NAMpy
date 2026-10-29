"""Weighted empirical measures, fixed strata and group-max exchangeability."""

import pickle
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nampy.calibration import (
    calibrate_conformal,
    calibrate_grouped,
    calibrate_regressor,
)
from tests.calibration.test_predictor import data


def reference_quantile(scores, weights, test, level):
    # Independent rational arithmetic, including the query's infinity mass.
    cumulative = Fraction(0)
    total = sum(map(Fraction, weights))+Fraction(test)
    for score, weight in sorted(zip(scores, weights, strict=True)):
        cumulative += Fraction(weight)
        if cumulative/total >= level:
            return score
    return np.inf


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_weighted_reference_infinity_scale_and_constant_weight_equivalence(method):
    y = np.arange(1., 10.)
    prediction = np.zeros(9) if method == "absolute" else np.column_stack([y-2, y+3])
    state = calibrate_conformal(y, prediction, method=method, level=.8, density_ratio=np.arange(1., 10.))
    query = np.zeros(3) if method == "absolute" else np.array([[-1, 1]]*3)
    ratios = np.array([1., 9., 100.])
    result = state.interval(query, density_ratio=ratios)
    expected = [reference_quantile(state.scores, list(range(1, 10)), int(w), Fraction(4, 5)) for w in ratios]
    assert_array_equal(result.correction, expected)
    scaled = calibrate_conformal(y, prediction, method=method, level=.8, density_ratio=np.arange(1., 10.)*1e200)
    assert_array_equal(scaled.interval(query, density_ratio=ratios*1e200).correction, expected)
    ordinary = calibrate_conformal(y, prediction, method=method, level=.9)
    uniform = calibrate_conformal(y, prediction, method=method, level=.9, density_ratio=np.ones(9))
    assert_array_equal(uniform.interval(query, density_ratio=np.ones(3)).correction, np.full(3, ordinary.correction))
    assert result.coverage == "marginal_under_known_covariate_shift"
    assert not result.correction.flags.writeable
    assert not pickle.loads(pickle.dumps(state)).density_ratio.flags.writeable


def test_weighted_finite_sample_coverage_by_exhaustive_target_assignment():
    # Conditional on the unordered augmented sample, the target index has
    # probability proportional to its density ratio (weighted exchangeability).
    scores, ratios = np.array([1., 2., 3., 4., 5.]), np.array([1., 2., 3., 4., 5.])
    covered_mass = 0
    for held in range(5):
        take = np.arange(5) != held
        state = calibrate_conformal(scores[take], np.zeros(4), level=.8, density_ratio=ratios[take])
        interval = state.interval([0.], density_ratio=ratios[[held]])
        covered_mass += ratios[held]*(scores[held] <= interval.upper[0])
    assert covered_mass/ratios.sum() >= .8


def test_weighting_unsupported_and_estimation_labels():
    with pytest.raises(ValueError, match="align"):
        calibrate_conformal([1, 2], [0, 0], density_ratio=[1])
    with pytest.raises(ValueError, match="nonnegative"):
        calibrate_conformal([1, 2], [0, 0], density_ratio=[-1, 1])
    state = calibrate_conformal([1, 2], [0, 0], density_ratio=[0, 0], weight_source="estimated")
    with pytest.raises(ValueError, match="test-point"):
        state.interval([0.])
    with pytest.raises(ValueError, match="positive"):
        state.interval([0.], density_ratio=[0.])
    result = state.interval([0.], density_ratio=[1.])
    assert np.isinf(result.upper[0]) and result.coverage == "approximate_with_estimated_density_ratio"


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_cluster_scores_use_one_maximum_per_group_and_new_group_coverage(method):
    y = np.array([1., 3., 2., 4., 6., 5.])
    groups = ["a", "a", "b", "c", "c", "c"]
    pred = np.zeros(6) if method == "absolute" else np.array([[-1., 1.]]*6)
    state = calibrate_grouped(y, pred, groups, level=.5, method=method)
    expected = [3, 2, 6] if method == "absolute" else [2, 1, 5]
    assert_array_equal(state.states[0].scores, expected)
    assert state.states[0].correction == sorted(expected)[1]
    query = np.zeros(2) if method == "absolute" else np.array([[-1., 1.]]*2)
    interval = state.interval(query, groups=["new", "new"])
    assert interval.n_calibration == 3
    assert interval.scope == "simultaneous_within_one_new_group"
    with pytest.raises(NotImplementedError, match="new group"):
        state.interval(query, groups=["a", "new"])


def test_cluster_exhaustive_holdout_covers_complete_group():
    groups = [np.array([1., 2.]), np.array([3.]), np.array([1., 4., 2.]), np.array([5.])]
    covered = 0
    for held in range(4):
        train = [i for i in range(4) if i != held]
        y = np.concatenate([groups[i] for i in train])
        ids = np.concatenate([[i]*len(groups[i]) for i in train])
        state = calibrate_grouped(y, np.zeros(len(y)), ids, level=.75)
        test = state.interval(np.zeros(len(groups[held])), groups=[held]*len(groups[held]))
        covered += bool(np.all(groups[held] <= test.upper))
    assert covered == 3


def test_fixed_strata_have_separate_counts_and_no_pooled_fallback():
    state = calibrate_grouped([1., 2., 3., 10.], np.zeros(4), ["a", "a", "a", "b"], mode="stratum", level=.5)
    result = state.interval([0., 0.], groups=["a", "b"])
    assert_array_equal(result.correction, [2., 10.])
    assert_array_equal(result.calibration_counts, [3, 1])
    assert result.coverage == "within_fixed_stratum_under_exchangeability"
    with pytest.raises(NotImplementedError, match="Unseen strata"):
        state.interval([0.], groups=["new"])
    sparse = calibrate_grouped([1.], [0.], ["a"], mode="stratum", level=.9)
    assert np.isinf(sparse.interval([0.], groups=["a"]).upper[0])


@pytest.mark.parametrize("backend", ["statistical", "neural"])
@pytest.mark.parametrize("kind", ["weighted", "cluster", "stratum"])
def test_estimator_wrappers_use_shared_weighted_grouped_state(kind, backend, tmp_path):
    from nampy import GAMRegressor
    from nampy.calibration import ConformalPredictor
    X, y = data()
    if backend == "statistical":
        model = GAMRegressor(formula="~x", optimize_smoothing=False).fit(X.iloc[:50], y[:50])
    else:
        from nampy import GradientTrainingConfig, LinRegRegressor
        model = LinRegRegressor(training=GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})).fit(X.iloc[:50], y[:50])
    cal_args = {"density_ratio": np.ones(25)} if kind == "weighted" else {
        "groups": np.repeat(np.arange(5), 5), "group_mode": kind}
    fitted = calibrate_regressor(model, X.iloc[50:75], y[50:75], level=.8, **cal_args)
    options = {"density_ratio": np.ones(15)} if kind == "weighted" else {
        "groups": np.full(15, 8 if kind == "cluster" else 0)}
    result = fitted.predict_interval(X.iloc[75:], **options)
    loaded = ConformalPredictor.load_model(fitted.save_model(tmp_path/f"{kind}.nampy"))
    assert_array_equal(loaded.predict_interval(X.iloc[75:], **options).lower, result.lower)
    with pytest.raises(NotImplementedError, match="Combined"):
        calibrate_regressor(model, X.iloc[50:75], y[50:75], groups=np.arange(25), density_ratio=np.ones(25))


@pytest.mark.parametrize("n,level", [(9, .8), (49, .14), (9, .9), (4, .6)])
def test_uniform_density_ratios_reduce_to_original_finite_sample_rank(n, level):
    unweighted = calibrate_conformal(np.arange(1., n+1), np.zeros(n), level=level)
    weighted = calibrate_conformal(np.arange(1., n+1), np.zeros(n), level=level, density_ratio=np.ones(n))
    assert_array_equal(weighted.interval([0.], density_ratio=[1.]).correction, [unweighted.correction])
