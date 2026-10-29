"""Whole-subject pairs bootstrap, backend refits, provenance and R references."""

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
from nampy.survival import SurvivalBootstrapError, SurvivalTarget, bootstrap_survival
from tests.survival.test_censored_fitting import sample
from tests.survival.test_censored_smoothing import formulas, smooth_sample


def reference_data():
    X = pd.DataFrame({"x": np.zeros(6)})
    target = SurvivalTarget([.3, .8, 1.4, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1],
                            [0, .2, .1, .4, 1, .5], subject_id=list("abcdef"))
    weight = np.array([1, 2, .5, 1, 0, 3])
    offset = np.array([.1, -.1, .03, -.04, .2, 0])
    model = SurvivalGAM(cuts=[0, 1, 4], formula="~1", optimize_smoothing=False).fit(
        X, target, sample_weight=weight, offset=offset)
    return model, X, target, weight, offset


def test_subject_bootstrap_matches_independent_r_glm_refits_and_percentiles():
    root = Path(__file__).parents[1]/"reference_fixtures"
    indices = pd.read_csv(root/"survival_bootstrap_indices.csv").to_numpy()
    reference = pd.read_csv(root/"survival_bootstrap.csv")
    model, X, target, weight, offset = reference_data()
    prediction_X = pd.DataFrame({"x": [0., 0.]})
    prediction_offset = np.array([-.05, .05])
    before = model.predict_link(X, offset=offset)
    result = bootstrap_survival(model, X, target, prediction_X, resample_indices=indices,
        sample_weight=weight, offset=offset, prediction_offset=prediction_offset, seed=21)
    assert result.training_policy == "not_applicable" and result.uncertainty_kind == "subject_bootstrap"
    assert not result.failures and result.successful_replicates == tuple(range(6))
    assert_array_equal(result.resample_indices, indices)
    assert_array_equal(result.source_subject_id, target.subject_id)
    assert not result.resample_indices.flags.writeable
    grid, landmark = [0, .5, 1, 2, 4], [0, .5]
    for b, value in [(-1, result.point_prediction), *enumerate(result.predictions)]:
        expected = reference[reference.replicate == b]
        assert_allclose(value.survival(grid, conditional_on=landmark).ravel(), expected.survival,
                        atol=2e-10, rtol=2e-10, equal_nan=True)
        assert_allclose(value.hazard_rates,
            np.exp(prediction_offset[:, None])*expected[["rate1", "rate2"]].iloc[0].to_numpy(), atol=2e-10)
    interval = result.percentile_interval(grid=grid, conditional_on=landmark)
    expected = reference[reference.replicate == -1]
    assert_allclose(interval.lower.ravel(), expected.lower, atol=2e-10, equal_nan=True)
    assert_allclose(interval.upper.ravel(), expected.upper, atol=2e-10, equal_nan=True)
    assert interval.coverage == "pointwise_percentile" and interval.uncertainty_kind == "subject_bootstrap"
    assert_array_equal(model.predict_link(X, offset=offset), before)
    for kind in ("hazard", "rmst"):
        functional = result.percentile_interval(grid=[.5, 1, 2], estimand=kind, conditional_on=landmark)
        if kind == "hazard":
            draws = np.stack([value.hazard([.5, 1, 2]) for value in result.predictions])
        else:
            draws = np.stack([np.column_stack([value.rmst(h, conditional_on=landmark) for h in [.5, 1, 2]]) for value in result.predictions])
        limits = np.quantile(draws, [.025, .975], axis=0)
        assert_allclose(functional.lower, limits[0])
        assert_allclose(functional.upper, limits[1])


def test_seeded_uniform_resampling_reproducibility_and_real_row_duplication(monkeypatch):
    model, X, target, weight, offset = reference_data()
    clones = []
    original = model.clone_for_survival_bootstrap
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        clones.append(value[0])
        return value
    monkeypatch.setattr(model, "clone_for_survival_bootstrap", capture)
    a = bootstrap_survival(model, X, target, X.iloc[:2], n_resamples=4, seed=19,
        offset=offset, prediction_offset=np.zeros(2), sample_weight=weight, on_failure="record")
    b = bootstrap_survival(model, X, target, X.iloc[:2], n_resamples=4, seed=19,
        offset=offset, prediction_offset=np.zeros(2), sample_weight=weight, on_failure="record")
    assert_array_equal(a.resample_indices, b.resample_indices)
    assert a.failures == b.failures
    assert all(replica is not model for replica in clones)
    for i, selected in enumerate(a.resample_indices):
        replica = clones[i]
        if i not in a.successful_replicates:
            continue
        independent = SurvivalGAM(cuts=[0, 1, 4], formula="~1", optimize_smoothing=False).fit(
            X.iloc[selected].reset_index(drop=True), SurvivalTarget(target.time[selected], target.event[selected], target.entry[selected]),
            sample_weight=weight[selected], offset=offset[selected])
        assert_array_equal(replica.predict_link(X.iloc[:2], offset=np.zeros(2)),
                           independent.predict_link(X.iloc[:2], offset=np.zeros(2)))


def test_failed_replicates_are_not_retried_or_silently_removed():
    model, X, target, weight, offset = reference_data()
    indices = np.stack([np.arange(6), np.full(6, 1), np.arange(5, -1, -1)])
    before = model.predict_link(X, offset=offset)
    kwargs = {"resample_indices": indices, "sample_weight": weight, "offset": offset,
              "prediction_offset": np.zeros(2)}
    with pytest.raises(SurvivalBootstrapError) as error:
        bootstrap_survival(model, X, target, X.iloc[:2], **kwargs)
    assert error.value.failure.replicate == 1 and error.value.failure.stage == "fit"
    result = bootstrap_survival(model, X, target, X.iloc[:2], on_failure="record", **kwargs)
    assert result.successful_replicates == (0, 2)
    assert len(result.failures) == 1 and result.failures[0].replicate == 1
    assert_array_equal(result.resample_indices, indices)
    with pytest.raises(ValueError, match="every requested replicate"):
        result.percentile_interval(grid=[0, 1])
    assert_array_equal(model.predict_link(X, offset=offset), before)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_censored_statistical_bootstrap_reuses_native_fitting(name):
    X, target = smooth_sample(name)
    model = SurvivalGAM(likelihood=name, formula=formulas(name, both=False)).fit(X, target)
    # Full-data and permuted-data index plans also exercise ordinary recompilation
    # and automatic REML without a fixed-design multiplicity-weight shortcut.
    indices = np.stack([np.arange(len(X)), np.arange(len(X)-1, -1, -1)])
    result = bootstrap_survival(model, X, target, X.iloc[:3], resample_indices=indices)
    assert len(result.predictions) == 2 and not result.failures
    for b, selected in enumerate(indices):
        direct = SurvivalGAM(likelihood=name, formula=formulas(name, both=False)).fit(
            X.iloc[selected].reset_index(drop=True),
            SurvivalTarget(target.time[selected], target.event[selected], target.entry[selected]))
        assert_array_equal(result.predictions[b].survival([0, 1, 2]), direct.predict_survival(X.iloc[:3], times=[0, 1, 2]))


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_duplicate_censored_subjects_and_saved_template_refit_identically(name, tmp_path):
    X, target, weight = sample(name)
    offset = np.column_stack([.02*X.x, -.01*X.x])
    model = SurvivalGAM(likelihood=name, formula="~x", optimize_smoothing=False).fit(
        X, target, sample_weight=weight, offset=offset)
    saved = SurvivalGAM.load_model(model.save_model(tmp_path/"bootstrap-source.nampy"))
    indices = np.stack([np.arange(len(X)), np.resize(np.arange(len(X)//2), len(X))])
    kwargs = {"resample_indices": indices, "sample_weight": weight, "offset": offset,
              "prediction_offset": offset[:2]}
    result = bootstrap_survival(model, X, target, X.iloc[:2], **kwargs)
    replay = bootstrap_survival(saved, X, target, X.iloc[:2], **kwargs)
    for i, selected in enumerate(indices):
        direct = SurvivalGAM(likelihood=name, formula="~x", optimize_smoothing=False).fit(
            X.iloc[selected].reset_index(drop=True),
            SurvivalTarget(target.time[selected], target.event[selected], target.entry[selected]),
            sample_weight=weight[selected], offset=offset[selected])
        expected = direct.predict_survival(X.iloc[:2], times=[0, 1, 2], offset=offset[:2])
        assert_array_equal(result.predictions[i].survival([0, 1, 2]), expected)
        assert_array_equal(replay.predictions[i].survival([0, 1, 2]), expected)


def neural_data(name):
    X, target, _ = sample("lognormal" if name == "lognormal" else "weibull")
    full = SurvivalTarget(target.time, target.event, target.entry,
                          subject_id=[f"person-{i}" for i in range(len(X))])
    n = len(X)-20
    return X.iloc[:n].reset_index(drop=True), full.take(np.arange(n)), X.iloc[n:].reset_index(drop=True), full.take(np.arange(n, len(X)))


@pytest.mark.parametrize("estimator_class", [LinRegSurvival, NAMSurvival])
@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
def test_neural_bootstrap_fresh_preprocessing_fixed_validation_and_seed_sources(name, estimator_class, tmp_path, monkeypatch):
    X, target, X_val, y_val = neural_data(name)
    extra = {}
    if estimator_class is NAMSurvival:
        from nampy.neural.configs.nam_config import DefaultNAMConfig
        extra["architecture_config"] = DefaultNAMConfig(layer_sizes=[4], dropout=0)
    training = GradientTrainingConfig(max_epochs=1, batch_size=128, random_state=17,
        checkpoint_path=str(tmp_path), trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})
    weight = np.resize([1., 0., 2., .5], len(X))
    offset = (np.linspace(-.03, .04, len(X)) if name == "piecewise_exponential"
              else np.column_stack([.03*X.x, -.02*X.x]))
    val_offset = np.zeros(len(X_val)) if offset.ndim == 1 else np.zeros((len(X_val), 2))
    model = estimator_class(objective=SurvivalObjectiveConfig(likelihood=name,
        cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None), training=training, **extra).fit(
            X, target, X_val=X_val, y_val=y_val, offset=offset, offset_val=val_offset, sample_weight=weight)
    indices = np.stack([np.arange(len(X)), np.resize(np.arange(len(X)//2), len(X))])
    captured = []
    fit_inputs = []
    original = model.clone_for_survival_bootstrap
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value[0])
        fit = value[0].fit
        def fit_and_record(features, response, **options):
            fit_inputs.append((features.copy(deep=True), response,
                               options["X_val"].copy(deep=True), options["y_val"],
                               options["offset"].copy(), options["sample_weight"].copy()))
            return fit(features, response, **options)
        value[0].fit = fit_and_record
        return value
    monkeypatch.setattr(model, "clone_for_survival_bootstrap", capture)
    before = model.predict_link(X.iloc[:3], offset=offset[:3])
    result = bootstrap_survival(model, X, target, X.iloc[:3], X_val=X_val, y_val=y_val,
        offset=offset, offset_val=val_offset, prediction_offset=offset[:3], sample_weight=weight,
        resample_indices=indices, seed=82)
    assert result.training_policy == "fixed_seed" and result.training_seeds == (17, 17)
    assert result.uncertainty_kind == "subject_bootstrap"
    assert result.validation_policy == "fixed_external"
    assert model.training is training
    for replica, selected, inputs in zip(captured, indices, fit_inputs, strict=True):
        assert replica.fitted_preprocessing_ is not model.fitted_preprocessing_
        pd.testing.assert_frame_equal(inputs[0], X.iloc[selected].reset_index(drop=True))
        pd.testing.assert_frame_equal(inputs[2], X_val)
        assert_array_equal(inputs[1].time, target.time[selected])
        assert_array_equal(inputs[1].entry, target.entry[selected])
        assert_array_equal(inputs[3].subject_id, y_val.subject_id)
        assert_array_equal(inputs[4], offset[selected])
        assert_array_equal(inputs[5], weight[selected])
        assert not Path(replica.training.checkpoint_path).exists()
    assert_array_equal(result.predictions[0].survival([0, 1, 2]), result.point_prediction.survival([0, 1, 2]))
    interval = result.percentile_interval(grid=[0, 1, 2])
    assert interval.coverage == "pointwise_percentile"
    assert_array_equal(before, model.predict_link(X.iloc[:3], offset=offset[:3]))
    if estimator_class is LinRegSurvival and name == "weibull":
        independent = bootstrap_survival(model, X, target, X.iloc[:3], X_val=X_val, y_val=y_val,
            offset=offset, offset_val=val_offset, prediction_offset=offset[:3], sample_weight=weight,
            resample_indices=indices, seed=82, training_randomness="independent")
        assert independent.training_seeds[0] != independent.training_seeds[1]
        assert independent.uncertainty_kind == "subject_bootstrap_with_training_randomness"
        assert_array_equal(independent.resample_indices, result.resample_indices)


def test_bootstrap_validation_and_unsupported_neural_routes(tmp_path):
    X, target, X_val, y_val = neural_data("weibull")
    training = GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path),
        trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})
    model = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood="weibull"), training=training).fit(
        X, target, X_val=X_val, y_val=y_val)
    with pytest.raises(NotImplementedError, match="external validation"):
        bootstrap_survival(model, X, target, X.iloc[:2], n_resamples=2)
    with pytest.raises(ValueError, match="overlap"):
        bootstrap_survival(model, X, target, X.iloc[:2], X_val=X.iloc[:2], y_val=target.take([0, 1]), n_resamples=2)
    with pytest.raises(ValueError, match="explicit source"):
        bootstrap_survival(model, X, SurvivalTarget(target.time, target.event, target.entry), X.iloc[:2],
            X_val=X_val, y_val=y_val, n_resamples=2)
    model.training = replace(training, warm_start=True)
    with pytest.raises(NotImplementedError, match="warm starts"):
        bootstrap_survival(model, X, target, X.iloc[:2], X_val=X_val, y_val=y_val, n_resamples=2)


def test_invalid_resampling_plans_and_interval_routes():
    model, X, target, weight, offset = reference_data()
    kwargs = {"sample_weight": weight, "offset": offset, "prediction_offset": np.zeros(2)}
    for indices in (np.zeros((2, 6), dtype=float), np.zeros((1, 6), dtype=int), np.full((2, 6), 6)):
        with pytest.raises(ValueError, match="integer"):
            bootstrap_survival(model, X, target, X.iloc[:2], resample_indices=indices, **kwargs)
    result = bootstrap_survival(model, X, target, X.iloc[:2], resample_indices=np.tile(np.arange(6), (2, 1)), **kwargs)
    with pytest.raises(NotImplementedError, match="survival, hazard or RMST"):
        result.percentile_interval(grid=[0, 1], estimand="effect")
    with pytest.raises(NotImplementedError, match="training randomness"):
        bootstrap_survival(model, X, target, X.iloc[:2], n_resamples=2, training_randomness="independent", **kwargs)


def test_reported_nonconvergence_is_a_failed_replicate(monkeypatch):
    from nampy.distributions.contracts import DistributionFitStatus

    model, X, target, weight, offset = reference_data()
    original = model.clone_for_survival_bootstrap
    # SurvivalGAM's fit transaction replaces its dictionary; patch the class
    # status only after each successful fit to preserve the real fit behavior.
    def configure(*args, **kwargs):
        replica, policy, seed = original(*args, **kwargs)
        fit = replica.fit
        def fit_with_failed_status(*a, **k):
            fit(*a, **k)
            replica.distribution_fit_status = lambda: DistributionFitStatus("gam", "nonconverged", "injected outer failure")
            return replica
        replica.fit = fit_with_failed_status
        return replica, policy, seed
    monkeypatch.setattr(model, "clone_for_survival_bootstrap", configure)
    result = bootstrap_survival(model, X, target, X.iloc[:2], sample_weight=weight, offset=offset,
        prediction_offset=np.zeros(2), resample_indices=np.tile(np.arange(6), (2, 1)), on_failure="record")
    assert len(result.predictions) == 0 and len(result.failures) == 2
    assert all(failure.stage == "fit_status" for failure in result.failures)
