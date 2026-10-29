"""Shared effects and pointwise uncertainty with predictor-coordinate checks."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.survival import (
    SurvivalPrediction,
    combine_survival_predictions,
    prepare_survival_curve_interval,
    prepare_survival_effects,
)
from tests.survival.test_censored_fitting import sample
from tests.survival.test_censored_smoothing import formulas, smooth_sample


@pytest.fixture(scope="module", params=["weibull", "lognormal", "piecewise_exponential"])
def statistical(request):
    name = request.param
    X, target, weight = sample("lognormal" if name == "lognormal" else "weibull")
    model = SurvivalGAM(likelihood=name, cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None,
                        formula="~x", optimize_smoothing=False).fit(X, target, sample_weight=weight)
    return model, X.iloc[:8], target, weight


def test_statistical_effects_use_reference_predictors_before_survival(statistical):
    model, X, _, _ = statistical
    reference = X.iloc[::2]
    weight = np.arange(1, len(reference)+1)
    offset = (np.linspace(-.1, .1, len(X)) if model.likelihood_ == "piecewise_exponential"
              else np.tile([.02, -.03], (len(X), 1)))
    ref_offset = (np.zeros(len(reference)) if offset.ndim == 1 else np.zeros((len(reference), 2)))
    prepared = prepare_survival_effects(model, X, times=[0, .5, 2], reference_X=reference,
        reference_weight=weight, offset=offset, reference_offset=ref_offset, conditional_on=.2)
    # Each model has one selected covariate term. Adding it back reconstructs
    # the full predictor; the interval-factor baseline must stay interval-specific.
    assert len(prepared.effects) == 1
    term = next(iter(prepared.effects))
    actual = model.predict_survival(X, times=[0, .5, 2], offset=offset, conditional_on=.2)
    assert_allclose(prepared.survival[term], actual, atol=1e-12)
    assert_allclose(prepared.effects[term], actual-prepared.reference_survival, equal_nan=True)
    assert np.isnan(prepared.reference_survival[:, 0]).all()
    assert prepared.features[term] == ("x",)
    assert_array_equal(prepared.feature_values[term][:, 0], X.x)
    assert not prepared.effects[term].flags.writeable
    with pytest.raises(TypeError):
        prepared.effects[term] = actual


def test_statistical_intervals_preserve_offsets_and_zero_draw_endpoints(statistical, tmp_path):
    model, X, _, _ = statistical
    p = len(model.gam_.fit_result().coef_full)
    offset = (np.full(len(X), .04) if model.likelihood_ == "piecewise_exponential"
              else np.tile([.02, -.03], (len(X), 1)))
    before = model.predict_link(X, offset=offset)
    result = prepare_survival_curve_interval(model, X, times=[0, .5, 2], offset=offset,
        standard_draws=np.zeros((2, p)), conditional_on=np.linspace(0, .3, len(X)))
    assert_allclose(result.lower, result.estimate, atol=0, rtol=0, equal_nan=True)
    assert_allclose(result.upper, result.estimate, atol=0, rtol=0, equal_nan=True)
    assert result.uncertainty_kind == "coefficient_conditional" and result.coverage == "pointwise"
    assert not result.lower.flags.writeable
    restored = SurvivalGAM.load_model(model.save_model(tmp_path / "survival.nampy"))
    z = np.random.default_rng(29).normal(size=(48, p))
    a = prepare_survival_curve_interval(model, X, times=[0, .5, 2], offset=offset, standard_draws=z)
    b = prepare_survival_curve_interval(restored, X, times=[0, .5, 2], offset=offset, standard_draws=z)
    assert_array_equal(a.lower, b.lower)
    assert_array_equal(a.upper, b.upper)
    assert np.all(a.lower <= a.upper) and np.all((a.lower >= 0) & (a.upper <= 1))
    assert_array_equal(before, model.predict_link(X, offset=offset))


def test_full_joint_covariance_transported_to_survival(statistical):
    model, X, _, _ = statistical
    core = model.gam_.gam_result_.fit_core_solution.fit_result
    beta, V = core.coef_full, core.cov_bayes
    z = np.random.default_rng(75).normal(size=(80, len(beta)))
    value, vector = np.linalg.eigh((V+V.T)/2)
    deviations = z @ ((vector*np.sqrt(np.maximum(value, 0))) @ vector.T)
    if model.likelihood_ == "piecewise_exponential":
        frame, _, _, _ = model._prediction_data(X, None)
        matrix = model.poisson_.lpmatrix(frame)
        eta = (matrix @ (beta+deviations).T).T.reshape(len(z), len(X), -1)
    else:
        matrix = model.parametric_.lpmatrix(X)
        blocks = model.gam_.gam_result_.require_compiled_model().predictor_full_indices
        eta = np.stack([(matrix[:, indices] @ (beta+deviations)[:, indices].T).T
                        for indices in blocks], axis=-1)
        # Censor-aware fitting couples predictors. Independent marginal eta
        # errors would discard these off-diagonal covariance blocks.
        assert np.max(np.abs(V[np.ix_(blocks[0], blocks[1])])) > 1e-8
    times = [0, .5, 1.5]
    curves = np.stack([model.survival_from_predictors(item).survival(times, conditional_on=.2) for item in eta])
    result = prepare_survival_curve_interval(model, X, times=times, standard_draws=z, conditional_on=.2)
    expected = np.quantile(curves[:, :, 1:], [.025, .975], axis=0)
    assert_allclose(result.lower[:, 1:], expected[0], atol=2e-14)
    assert_allclose(result.upper[:, 1:], expected[1], atol=2e-14)


@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
def test_neural_effects_and_explicit_uncertainty_limit(name, tmp_path):
    X, target, _ = sample("lognormal" if name == "lognormal" else "weibull")
    model = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood=name,
        cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None),
        training=GradientTrainingConfig(max_epochs=1, batch_size=128, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False}),
    ).fit(X, target)
    X = X.iloc[:6]
    effect = prepare_survival_effects(model, X, times=[0, 1, 2])
    assert len(effect.effects) == 1
    curve = model.predict_distribution(X)
    assert_allclose(next(iter(effect.survival.values())), curve.survival([0, 1, 2]), atol=2e-7)
    if name == "piecewise_exponential":
        known = np.full(len(X), 1e-9, dtype=np.float64)
        shifted = prepare_survival_effects(model, X, times=[0, 1, 2], offset=known)
        assert_allclose(shifted.reference_survival,
                        np.exp(np.log(effect.reference_survival)*np.exp(known[:, None])),
                        atol=3e-15, rtol=0)
    with pytest.raises(NotImplementedError, match="Neural survival coefficient"):
        prepare_survival_curve_interval(model, X, times=[0, 1, 2])
    repeat = model.survival_from_predictors(model.predict_link(X)+.1)
    combined = combine_survival_predictions([curve, repeat], times=[0, 1, 2])
    assert combined.uncertainty_kind == "between_fits" and combined.coverage == "empirical_range"
    assert_allclose(combined.estimate, (curve.survival([0, 1, 2])+repeat.survival([0, 1, 2]))/2)
    assert not np.array_equal(combined.estimate, model.survival_from_predictors(
        model.predict_link(X)+.05).survival([0, 1, 2]))


def test_shared_interval_contract_rejects_bad_routes_and_preserves_landmark_mask(statistical):
    model, X, _, _ = statistical
    curve = model.predict_distribution(X)
    with pytest.raises(ValueError, match="two"):
        combine_survival_predictions([curve], times=[0, 1])
    with pytest.raises(ValueError, match="rows"):
        combine_survival_predictions([curve, model.predict_distribution(X.iloc[:2])], times=[0, 1])
    with pytest.raises(NotImplementedError, match="covariance"):
        prepare_survival_curve_interval(model, X, times=[0, 1], covariance="freq")
    with pytest.raises(NotImplementedError, match="Unavailable"):
        prepare_survival_effects(model, X, times=[0, 1], terms=["absent"])
    combined = combine_survival_predictions([curve, curve], times=[0, .5, 1], conditional_on=.5)
    assert np.isnan(combined.lower[:, 0]).all()
    assert_array_equal(combined.lower[:, 1], np.ones(len(X)))
    assert_array_equal(combined.lower[:, 1:], combined.estimate[:, 1:])


def test_shared_aggregation_averages_probabilities_not_hazards():
    a = SurvivalPrediction(np.array([[.1], [.2]]), [0, 2])
    b = replace(a, hazard_rates=np.array([[1.], [.8]]))
    result = combine_survival_predictions([a, b], times=[0, 1, 2], conditional_on=[0, 1])
    assert_allclose(result.estimate, .5*(a.survival([0, 1, 2], conditional_on=[0, 1])
                                      + b.survival([0, 1, 2], conditional_on=[0, 1])), equal_nan=True)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_available_smoothing_corrected_covariance_with_two_smooth_predictors(name, tmp_path):
    X, target = smooth_sample(name)
    model = SurvivalGAM(likelihood=name, formula=formulas(name)).fit(X, target)
    p = len(model.gam_.fit_result().coef_full)
    z = np.random.default_rng(12).normal(size=(32, p))
    assert model.gam_.fit_result().core.cov_unconditional is not None
    a = prepare_survival_curve_interval(model, X.iloc[:4], times=[0, 1, 2],
                                        covariance="unconditional", standard_draws=z)
    b = prepare_survival_curve_interval(model, X.iloc[:4], times=[0, 1, 2], standard_draws=z)
    assert a.uncertainty_kind == "coefficient_smoothing_corrected"
    assert not np.array_equal(a.lower[:, 1:], b.lower[:, 1:])
    restored = SurvivalGAM.load_model(model.save_model(tmp_path / "corrected.nampy"))
    c = prepare_survival_curve_interval(restored, X.iloc[:4], times=[0, 1, 2],
                                        covariance="unconditional", standard_draws=z)
    assert_array_equal(a.lower, c.lower)
    effects = prepare_survival_effects(model, X.iloc[:4], times=[0, 1, 2])
    assert len(effects.effects) == 2


def test_missing_corrected_covariance_and_single_draw_do_not_substitute(statistical, monkeypatch):
    model, X, _, _ = statistical
    core = model.gam_.gam_result_.fit_core_solution
    result = replace(model.gam_.gam_result_, fit_core_solution=replace(
        core, fit_result=replace(core.fit_result, cov_unconditional=None)))
    monkeypatch.setattr(model.gam_, "gam_result_", result)
    with pytest.raises(ValueError, match="unavailable"):
        prepare_survival_curve_interval(model, X, times=[0, 1], covariance="unconditional")
    with pytest.raises(ValueError, match="two"):
        prepare_survival_curve_interval(model, X, times=[0, 1], n_draws=1)


def test_interaction_and_categorical_term_coordinates_are_preserved():
    import pandas as pd

    from nampy.contracts import AdditivePrediction, AdditiveTermMetadata

    class Adapter:
        def predict_components(self, X, offset=None):
            terms = {"x": np.column_stack([X.x, X.x*.5]),
                     "x:z": np.column_stack([X.x*X.z, -X.x*X.z]),
                     "group": np.column_stack([X.group.eq("b")*.2, X.group.eq("b")*.3])}
            intercept = np.array([-.6, -.2])
            return AdditivePrediction(link=intercept+sum(terms.values()), terms=terms,
                intercept=intercept, predictor_names=("first", "second"), backend="gam",
                term_metadata={"x": AdditiveTermMetadata(features=("x",)),
                    "x:z": AdditiveTermMetadata(features=("x", "z"), kind="interaction"),
                    "group": AdditiveTermMetadata(features=("group",))})

        def survival_from_predictors(self, eta):
            return SurvivalPrediction(np.exp(eta), [0, 1, 2])

    X = pd.DataFrame({"x": [-.2, .1, .4], "z": [.3, -.1, .2], "group": ["a", "b", "b"]})
    result = prepare_survival_effects(Adapter(), X, times=[0, .5, 2], reference_weight=[1, 2, 3])
    xmean = np.average(X.x, weights=[1, 2, 3])
    xymean = np.average(X.x*X.z, weights=[1, 2, 3])
    groupmean = np.average(X.group.eq("b"), weights=[1, 2, 3])
    baseline = [-.6+xmean+xymean+.2*groupmean, -.2+.5*xmean-xymean+.3*groupmean]
    centered_xy = X.x*X.z-xymean
    eta = np.asarray(baseline)+np.column_stack([centered_xy, -centered_xy])
    expected = np.exp(-np.exp(eta) @ np.array([[0, .5, 1], [0, 0, 1]]))
    assert_allclose(result.survival["x:z"], expected)
    assert result.feature_values["x:z"].shape == (3, 2)
    assert_array_equal(result.feature_values["group"][:, 0], X.group)


def test_piecewise_effects_follow_explicit_row_mapping(monkeypatch):
    from nampy.survival import SurvivalComponents

    X, target, _ = sample("weibull")
    model = SurvivalGAM(cuts=[0, .4, 1.2, 3], formula="~x", optimize_smoothing=False).fit(X, target)
    X = X.iloc[:5]
    expected = prepare_survival_effects(model, X, times=[0, 1, 2])
    source = model.predict_components(X)
    order = np.random.default_rng(7).permutation(len(source.subject_index))
    p = source.prediction
    shuffled = replace(p, link=p.link[order], terms={key: value[order] for key, value in p.terms.items()},
                       intercept=np.broadcast_to(p.intercept, p.link.shape)[order],
                       offset=None if p.offset is None else p.offset[order])
    record = SurvivalComponents(shuffled, source.subject_index[order], source.interval[order], source.evaluation_time[order])
    monkeypatch.setattr(model, "predict_components", lambda *a, **k: record)
    actual = prepare_survival_effects(model, X, times=[0, 1, 2])
    for term in expected.effects:
        assert_array_equal(actual.effects[term], expected.effects[term])


def test_nonidentity_prediction_coordinates_are_explicitly_rejected(statistical, monkeypatch):
    model, X, _, _ = statistical
    result = model.gam_.gam_result_
    compiled = result.require_compiled_model()
    p = len(result.fit_core_solution.fit_result.coef_full)
    changed = replace(compiled, fit_to_prediction_parameterization_map=2*np.eye(p))
    monkeypatch.setattr(model.gam_, "gam_result_", replace(result, compiled_model=changed))
    with pytest.raises(NotImplementedError, match="prediction coordinates"):
        prepare_survival_curve_interval(model, X, times=[0, 1], n_draws=2)


def test_missing_survival_adapters_are_explicit():
    with pytest.raises(NotImplementedError, match="predictor-evaluation adapter"):
        prepare_survival_effects(object(), np.zeros((2, 1)), times=[0, 1])
    with pytest.raises(NotImplementedError, match="coefficient-simulation adapter"):
        prepare_survival_curve_interval(object(), np.zeros((2, 1)), times=[0, 1])
