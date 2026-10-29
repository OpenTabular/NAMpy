"""Observed-time lower sets and administrative-cutoff weighted lower bounds."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nampy.calibration import calibrate_survival, calibrate_survival_estimator
from nampy.survival import SurvivalTarget
from tests.calibration.test_weighted_grouped import reference_quantile


def test_observed_time_scores_include_censored_records_and_cover_latent_times():
    latent = np.array([1., 2., 3., 4., 5.])
    censor = np.array([4., 1., 5., 2., 6.])
    covered = 0
    for held in range(5):
        take = np.arange(5) != held
        target = SurvivalTarget(np.minimum(latent[take], censor[take]), latent[take] <= censor[take])
        state = calibrate_survival(target, np.full(4, 3.), level=.8)
        interval = state.interval([3.])
        assert_array_equal(state.calibration.scores, 3-target.time)
        covered += bool(latent[held] >= interval.lower[0])
        assert np.isposinf(interval.upper[0])
        assert interval.coverage == "conservative_marginal_for_event_time"
    assert covered/5 >= .8


def test_cutoff_known_selection_matches_rational_weighted_reference():
    from fractions import Fraction
    latent = np.array([.5, 2., 5., 2.5, 4., 1.])
    censor = np.array([4., 1., 4., 4., 5., .8])
    target = SurvivalTarget(np.minimum(latent, censor), latent <= censor, subject_id=list("abcdef"))
    probability = np.array([1., .5, .5, .25, 1., .8])
    state = calibrate_survival(target, np.full(6, 2.), cutoff=3., censoring_time=censor,
                              selection_probability=probability, level=.5)
    # Selected a,c,d,e; clipped event times .5,3,2.5,3; ratios 1,2,4,1.
    assert_array_equal(state.calibration.scores, [1.5, -1, -.5, -1])
    assert_array_equal(state.selected_subject_id, list("acde"))
    interval = state.interval([2., 2.], selection_probability=[.5, .001])
    correction = reference_quantile([1.5, -1., -.5, -1.], [1, 2, 4, 1], 2, Fraction(1, 2))
    assert_array_equal(interval.correction, [correction, np.inf])
    assert_array_equal(interval.lower, [2-correction, 0])
    assert interval.coverage == "marginal_with_known_censoring_selection"
    estimated = calibrate_survival(target, np.full(6, 2.), cutoff=3., censoring_time=censor,
                                  selection_probability=probability, weight_source="estimated", level=.5)
    assert estimated.interval([2.], selection_probability=[.5]).coverage == "approximate_with_estimated_censoring_selection"


def test_survival_rejects_missing_censor_times_entry_and_invalid_probabilities():
    target = SurvivalTarget([1., 2.], [1, 0])
    with pytest.raises(ValueError, match="every subject"):
        calibrate_survival(target, [1., 1.], cutoff=2.)
    with pytest.raises(ValueError, match="agree"):
        calibrate_survival(target, [1., 1.], cutoff=2., censoring_time=[.5, 3.], selection_probability=[1., 1.])
    with pytest.raises(ValueError, match="positivity"):
        calibrate_survival(target, [1., 1.], cutoff=2., censoring_time=[2., 2.], selection_probability=[0., 1.])
    with pytest.raises(NotImplementedError, match="Delayed-entry"):
        calibrate_survival(SurvivalTarget([1., 2.], [1, 0], entry=[.1, 0]), [1., 1.])


@pytest.mark.parametrize("backend", ["statistical", "neural", "nam"])
@pytest.mark.parametrize("likelihood", ["piecewise_exponential", "weibull", "lognormal"])
def test_shared_survival_predictions_snapshot_bounds_and_persistence(backend, likelihood, tmp_path):
    import pandas as pd

    from nampy import (
        GradientTrainingConfig,
        LinRegSurvival,
        NAMSurvival,
        SurvivalGAM,
        SurvivalObjectiveConfig,
    )
    from nampy.calibration import SurvivalConformalPredictor
    rng = np.random.default_rng(818)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 90)})
    latent = np.exp(.2+.2*X.x+rng.normal(size=90)*.3)
    censor = np.full(90, 2.5)
    target = SurvivalTarget(np.minimum(latent, censor), latent <= censor)
    if backend == "statistical":
        model = SurvivalGAM(likelihood=likelihood, cuts=[0, 1, 3] if likelihood == "piecewise_exponential" else None,
                            formula="~x", optimize_smoothing=False)
    else:
        extra = {}
        if backend == "nam":
            from nampy.neural.configs.nam_config import DefaultNAMConfig
            extra["architecture_config"] = DefaultNAMConfig(layer_sizes=[4], dropout=0)
        cls = NAMSurvival if backend == "nam" else LinRegSurvival
        model = cls(**extra, objective=SurvivalObjectiveConfig(likelihood=likelihood,
            cuts=[0, 1, 3] if likelihood == "piecewise_exponential" else None),
            training=GradientTrainingConfig(max_epochs=1, random_state=18, checkpoint_path=str(tmp_path),
                trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False}))
    model.fit(X.iloc[:50], target.take(np.arange(50)))
    rows = np.arange(50, 75)
    before = model.predict_distribution(X.iloc[75:]).survival([.5, 1.5])
    for cutoff in (None, 2.):
        options = {} if cutoff is None else {"cutoff": cutoff, "censoring_time": censor[rows],
                                             "selection_probability": np.ones(25)}
        wrapper = calibrate_survival_estimator(model, X.iloc[rows], target.take(rows), level=.8, **options)
        prediction_options = {} if cutoff is None else {"selection_probability": np.ones(15)}
        result = wrapper.predict_interval(X.iloc[75:], **prediction_options)
        explicit = wrapper.calibration.interval(model.predict_distribution(X.iloc[75:]), **prediction_options)
        assert_array_equal(result.lower, explicit.lower)
        assert np.all(result.lower >= 0) and np.isposinf(result.upper).all()
        if cutoff is not None:
            assert np.all(result.lower <= cutoff)
        loaded = SurvivalConformalPredictor.load_model(wrapper.save_model(tmp_path/f"{backend}-{likelihood}-{cutoff}.nampy"))
        assert_array_equal(loaded.predict_interval(X.iloc[75:], **prediction_options).lower, result.lower)
        assert not loaded.calibration.selected_subject_id.flags.writeable
    assert_array_equal(model.predict_distribution(X.iloc[75:]).survival([.5, 1.5]), before)
