"""R references, paired-draw effects and finite-grid survival calibration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.contracts import AdditivePrediction, AdditiveTermMetadata
from nampy.gam.inference.functionals import coefficient_draws
from nampy.survival import (
    SurvivalEffectSamples,
    SurvivalPrediction,
    prepare_survival_curve_interval,
    prepare_survival_effect_interval,
)
from nampy.survival.likelihood import parametric_distribution
from tests.survival.test_censored_fitting import sample
from tests.survival.test_censored_smoothing import formulas, smooth_sample


class ReferenceAdapter:
    """Known two-predictor coefficient law, separate from any fitting kernel."""

    beta = np.array([.2, .35, -.1])
    covariance = np.array([[.04, .012, -.008], [.012, .03, .006], [-.008, .006, .02]])

    def __init__(self, name):
        self.name = name
        self.model = SimpleNamespace(_fitted=True, gam_result_=SimpleNamespace(
            fit_core_solution=SimpleNamespace(fit_result=SimpleNamespace(coef_full=self.beta))),
            _select_cov=lambda covariance: self.covariance)

    def eta(self, X, beta, offset=None):
        return np.column_stack([beta[0]+beta[1]*X.x, np.full(len(X), beta[2])])+(
            0 if offset is None else offset)

    def survival_from_predictors(self, eta):
        natural = (np.exp(eta) if self.name == "weibull" else
                   np.column_stack([eta[:, 0], np.exp(eta[:, 1])]))
        return SurvivalPrediction(distribution=parametric_distribution(self.name, natural))

    def predict_distribution(self, X, offset=None):
        return self.survival_from_predictors(self.eta(X, self.beta, offset))

    def predict_components(self, X, offset=None):
        term = np.column_stack([self.beta[1]*X.x, np.zeros(len(X))])
        return AdditivePrediction(link=self.eta(X, self.beta, offset), terms={"x": term},
            intercept=[self.beta[0], self.beta[2]], offset=offset, backend="gam",
            predictor_names=("first", "second"), term_metadata={"x": AdditiveTermMetadata(features=("x",))})

    def sample_survival_predictions(self, X, *, offset=None, **kwargs):
        return tuple(self.survival_from_predictors(self.eta(X, b, offset))
                     for b in coefficient_draws(self.model, **kwargs))

    def sample_survival_effect_predictions(self, X, *, terms, reference_X=None,
        reference_weight=None, offset=None, reference_offset=None, **kwargs):
        ref = X if reference_X is None else reference_X
        center = np.average(ref.x, weights=reference_weight)
        draws = coefficient_draws(self.model, **kwargs)
        baseline = pd.DataFrame({"x": np.full(len(X), center)})
        return SurvivalEffectSamples(
            tuple(self.survival_from_predictors(self.eta(baseline, b, offset)) for b in draws),
            {"x": tuple(self.survival_from_predictors(self.eta(X, b, offset)) for b in draws)})


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_simultaneous_curves_and_paired_effects_against_independent_r(name):
    expected = pd.read_csv(Path(__file__).parents[1]/"reference_fixtures/survival_inference.csv")
    expected = expected[expected.family == name]
    X = pd.DataFrame({"x": [-.5, .25, .8]})
    ref = pd.DataFrame({"x": [-.8, .1, .6, .95]})
    offsets = np.column_stack([.03*X.x, -.02*X.x])
    z = norm.ppf(((np.arange(1, 121)*7) % 113+.5)/113).reshape(40, 3)
    adapter = ReferenceAdapter(name)
    kwargs = {"times": [0, .3, 1, 2], "offset": offsets, "conditional_on": [0, .2, .3], "standard_draws": z}
    band = prepare_survival_curve_interval(adapter, X, interval="simultaneous", **kwargs)
    effect = prepare_survival_effect_interval(adapter, X, reference_X=ref,
        reference_weight=[1, 3, 2, 1], **kwargs).intervals["x"]
    simultaneous = prepare_survival_effect_interval(adapter, X, reference_X=ref,
        reference_weight=[1, 3, 2, 1], interval="simultaneous", **kwargs).intervals["x"]
    for result, fields in [(band, {"estimate": "estimate", "lower": "lower", "upper": "upper", "standard_error": "se"}),
                           (effect, {"estimate": "effect", "lower": "effect_lower", "upper": "effect_upper"}),
                           (simultaneous, {"estimate": "effect", "lower": "effect_sim_lower", "upper": "effect_sim_upper", "standard_error": "effect_se"})]:
        for actual, column in fields.items():
            assert_allclose(getattr(result, actual).ravel(), expected[column], atol=3e-13, rtol=3e-13, equal_nan=True)
    assert_allclose(band.critical_value, expected.critical.iloc[0], atol=3e-13)
    assert_allclose(simultaneous.critical_value, expected.effect_critical.iloc[0], atol=3e-13)
    assert band.scope == "all_subjects_and_times" and band.transformation == "log_cumulative_hazard"
    assert simultaneous.scope == "all_selected_terms_subjects_and_times"
    assert effect.estimand == "reference_plus_term_survival_difference"
    assert not simultaneous.lower.flags.writeable


@pytest.fixture(scope="module", params=["weibull", "lognormal", "piecewise_exponential"])
def fitted(request):
    name = request.param
    X, target, weights = sample("lognormal" if name == "lognormal" else "weibull")
    model = SurvivalGAM(likelihood=name, cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None,
                        formula="~x", optimize_smoothing=False).fit(X, target, sample_weight=weights)
    return model, X.iloc[:7]


@pytest.mark.parametrize("interval", ["pointwise", "simultaneous"])
def test_fitted_effect_draws_share_reference_covariance_and_survive_persistence(fitted, interval, tmp_path):
    model, X = fitted
    reference = X.iloc[[0, 2, 5]]
    weight = [1, 3, 2]
    offset = (np.linspace(-.1, .1, len(X)) if model.likelihood_ == "piecewise_exponential"
              else np.tile([.02, -.03], (len(X), 1)))
    ref_offset = np.zeros(len(reference)) if offset.ndim == 1 else np.zeros((len(reference), 2))
    p = len(model.gam_.fit_result().coef_full)
    z = np.random.default_rng(38).normal(size=(128, p))
    kwargs = {"times": [0, .5, 1.5, 2.5], "reference_X": reference, "reference_weight": weight,
              "offset": offset, "reference_offset": ref_offset, "conditional_on": .2, "standard_draws": z, "interval": interval}
    before = model.predict_link(X, offset=offset)
    result = prepare_survival_effect_interval(model, X, **kwargs)
    term = next(iter(result.intervals))
    assert len(result.intervals) == 1
    # With one feature, reference-plus-term restores the full fitted model in
    # every draw; baseline is the same model at the weighted reference x.
    centered_X = X.copy()
    centered_X["x"] = np.average(reference.x, weights=weight)
    original = model.sample_survival_predictions(X, offset=offset, standard_draws=z)
    baseline = model.sample_survival_predictions(centered_X, offset=offset, standard_draws=z)
    curves = np.stack([a.survival(kwargs["times"], conditional_on=.2)-b.survival(kwargs["times"], conditional_on=.2)
                       for a, b in zip(original, baseline, strict=True)])
    actual = result.intervals[term]
    if interval == "pointwise":
        expected = np.quantile(curves[:, :, 1:], [.025, .975], axis=0)
        assert_allclose(actual.lower[:, 1:], expected[0], atol=2e-14)
        assert_allclose(actual.upper[:, 1:], expected[1], atol=2e-14)
        # Independent-draw subtraction would change uncertainty substantially.
        independent = np.stack([a.survival(kwargs["times"], conditional_on=.2)-baseline[-i-1].survival(kwargs["times"], conditional_on=.2)
                                for i, a in enumerate(original)])
        assert np.max(np.abs(np.std(curves[:, :, 1:], axis=0)-np.std(independent[:, :, 1:], axis=0))) > 1e-4
    else:
        inside = (curves[:, :, 1:] >= actual.lower[:, 1:]-1e-15) & (curves[:, :, 1:] <= actual.upper[:, 1:]+1e-15)
        assert np.mean(np.all(inside, axis=(1, 2))) >= .95
    restored = SurvivalGAM.load_model(model.save_model(tmp_path/"effects.nampy"))
    replay = prepare_survival_effect_interval(restored, X, **kwargs).intervals[term]
    assert_array_equal(replay.lower, actual.lower)
    assert_array_equal(replay.upper, actual.upper)
    assert_array_equal(before, model.predict_link(X, offset=offset))


def test_finite_grid_band_coverage_boundaries_and_zero_draws(fitted):
    model, X = fitted
    p = len(model.gam_.fit_result().coef_full)
    z = np.random.default_rng(18).normal(size=(128, p))
    times = [0, .2, .5, 1, 2]
    band = prepare_survival_curve_interval(model, X, times=times, standard_draws=z,
                                           conditional_on=.2, interval="simultaneous")
    samples = model.sample_survival_predictions(X, standard_draws=z)
    curves = np.stack([value.survival(times, conditional_on=.2) for value in samples])
    inside = (curves[:, :, 1:] >= band.lower[:, 1:]-1e-15) & (curves[:, :, 1:] <= band.upper[:, 1:]+1e-15)
    assert np.mean(np.all(inside, axis=(1, 2))) >= .95
    assert np.isnan(band.lower[:, 0]).all()
    assert_array_equal(band.lower[:, 1], np.ones(len(X)))
    zero = prepare_survival_curve_interval(model, X, times=times, standard_draws=np.zeros((2, p)), interval="simultaneous")
    assert zero.critical_value == 0
    assert_allclose(zero.lower, zero.estimate, atol=2e-16)
    effect = prepare_survival_effect_interval(model, X, times=times,
        standard_draws=np.zeros((2, p)), interval="simultaneous")
    for value in effect.intervals.values():
        assert value.critical_value == 0
        assert_array_equal(value.lower, value.estimate)
    with pytest.raises(ValueError, match="valid grid"):
        prepare_survival_curve_interval(model, X, times=[0], conditional_on=.1,
                                         standard_draws=z, interval="simultaneous")


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_corrected_covariance_and_joint_term_scope(name):
    X, target = smooth_sample(name)
    model = SurvivalGAM(likelihood=name, formula=formulas(name)).fit(X, target)
    result = prepare_survival_effect_interval(model, X.iloc[:5], times=[0, .5, 1.5],
        covariance="unconditional", n_draws=64, seed=19, interval="simultaneous")
    assert len(result.intervals) == 2
    assert len({value.critical_value for value in result.intervals.values()}) == 1
    assert all(value.uncertainty_kind == "coefficient_smoothing_corrected" for value in result.intervals.values())
    curve = prepare_survival_curve_interval(model, X.iloc[:5], times=[0, .5, 1.5],
        covariance="unconditional", n_draws=64, seed=19, interval="simultaneous")
    assert np.isfinite(curve.lower).all()


def test_neural_simultaneous_and_effect_coefficient_routes_stay_unsupported(tmp_path):
    X, target, _ = sample("weibull")
    model = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood="weibull"),
        training=GradientTrainingConfig(max_epochs=1, batch_size=128, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})).fit(X, target)
    with pytest.raises(NotImplementedError, match="Neural survival coefficient"):
        prepare_survival_curve_interval(model, X, times=[0, 1], interval="simultaneous")
    with pytest.raises(NotImplementedError, match="Neural survival coefficient effect"):
        prepare_survival_effect_interval(model, X, times=[0, 1], interval="simultaneous")


@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
def test_grouped_categories_and_interactions_use_the_existing_formula_term_groups(name):
    X, target, _ = sample("lognormal" if name == "lognormal" else "weibull")
    X = X.assign(z=np.sin(np.arange(len(X))*.3), group=np.resize(["a", "b", "c"], len(X)))
    model = SurvivalGAM(likelihood=name, cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None,
                        formula="~x+factor(group)+x:z", optimize_smoothing=False).fit(X, target)
    p = len(model.gam_.fit_result().coef_full)
    result = prepare_survival_effect_interval(model, X.iloc[:6], times=[0, 1, 2],
        reference_X=X.iloc[10:20], reference_weight=np.arange(1, 11),
        standard_draws=np.zeros((2, p)), interval="simultaneous")
    assert len(result.intervals) == 3
    assert any(features == ("group",) for features in result.effects.features.values())
    assert any(set(features) == {"x", "z"} for features in result.effects.features.values())
    for value in result.intervals.values():
        assert_array_equal(value.estimate, value.lower)
        assert_array_equal(value.estimate, value.upper)


def test_effect_interval_failures_do_not_fall_back(fitted, monkeypatch):
    from dataclasses import replace

    model, X = fitted
    result = model.gam_.gam_result_
    core = result.fit_core_solution
    monkeypatch.setattr(model.gam_, "gam_result_", replace(result,
        fit_core_solution=replace(core, fit_result=replace(core.fit_result, cov_unconditional=None))))
    with pytest.raises(ValueError, match="unavailable"):
        prepare_survival_effect_interval(model, X, times=[0, 1], covariance="unconditional")
    with pytest.raises(NotImplementedError, match="covariance"):
        prepare_survival_effect_interval(model, X, times=[0, 1], covariance="freq")
    with pytest.raises(ValueError, match="pointwise/simultaneous"):
        prepare_survival_effect_interval(model, X, times=[0, 1], interval="continuous")


def test_tiny_tail_contrasts_retain_nonzero_sample_standard_deviation():
    from nampy.survival.uncertainty import _simultaneous_limits

    point = np.array([[2e-200, 1.]])
    draws = np.array([[[j*1e-200, 1.]] for j in range(5)])
    lower, upper, se, critical = _simultaneous_limits(point, draws, np.ones_like(point, dtype=bool), .95)
    assert_allclose(se[0, 0]/1e-200, np.sqrt(2.5), atol=1e-14)
    assert_allclose(critical, 2/np.sqrt(2.5), atol=1e-14)
    assert_allclose(lower[0, 0]/1e-200, 0, atol=1e-14)
    assert_allclose(upper[0, 0]/1e-200, 4, atol=1e-14)
    assert se[0, 1] == 0 and lower[0, 1] == upper[0, 1] == 1
