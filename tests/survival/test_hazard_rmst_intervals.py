"""Independent R references and fitted hazard/RMST coefficient inference."""

from dataclasses import replace
from pathlib import Path

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
from nampy.survival import (
    SurvivalPrediction,
    prepare_survival_hazard_interval,
    prepare_survival_rmst_interval,
)
from nampy.survival.likelihood import parametric_distribution
from tests.survival.test_censored_fitting import sample
from tests.survival.test_censored_smoothing import formulas, smooth_sample
from tests.survival.test_simultaneous_effect_intervals import ReferenceAdapter


class FunctionalReferenceAdapter(ReferenceAdapter):
    def eta(self, X, beta, offset=None):
        if self.name != "piecewise_exponential":
            return super().eta(X, beta, offset)
        first = beta[0]+beta[1]*X.x
        return np.column_stack([first, first+beta[2]])+(0 if offset is None else offset)

    def survival_from_predictors(self, eta):
        if self.name == "piecewise_exponential":
            return SurvivalPrediction(np.exp(eta), [0, 1, 3])
        return super().survival_from_predictors(eta)


@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
@pytest.mark.parametrize("kind", ["hazard", "rmst"])
def test_functional_intervals_against_independent_r_reference(name, kind):
    table = pd.read_csv(Path(__file__).parents[1]/"reference_fixtures/survival_functional_inference.csv")
    table = table[(table.family == name) & (table.kind == kind)]
    X = pd.DataFrame({"x": [-.5, .25, .8]})
    offset = np.column_stack([.03*X.x, -.02*X.x])
    z = norm.ppf(((np.arange(1, 121)*7) % 113+.5)/113).reshape(40, 3)
    function = prepare_survival_hazard_interval if kind == "hazard" else prepare_survival_rmst_interval
    argument = {"times": [.2, .5, 1, 2]} if kind == "hazard" else {"horizons": [0, .5, 1, 2.5]}
    kwargs = {**argument, "offset": offset, "conditional_on": [0, .5, 1], "standard_draws": z}
    pointwise = function(FunctionalReferenceAdapter(name), X, **kwargs)
    simultaneous = function(FunctionalReferenceAdapter(name), X, interval="simultaneous", **kwargs)
    tolerance = 3e-12 if kind == "hazard" else 2e-8
    for result, columns in [(pointwise, {"estimate": "estimate", "lower": "lower", "upper": "upper"}),
                            (simultaneous, {"estimate": "estimate", "lower": "sim_lower", "upper": "sim_upper", "standard_error": "se"})]:
        for attr, column in columns.items():
            assert_allclose(getattr(result, attr).ravel(), table[column], atol=tolerance, rtol=tolerance, equal_nan=True)
    assert_allclose(simultaneous.critical_value, table.critical.iloc[0], atol=tolerance, rtol=tolerance)
    assert simultaneous.scope == ("all_subjects_and_times" if kind == "hazard" else "all_subjects_and_horizons")
    assert simultaneous.transformation == ("log_hazard" if kind == "hazard" else "rmst")
    assert simultaneous.estimand == ("hazard" if kind == "hazard" else "restricted_mean_remaining_lifetime")
    assert not simultaneous.estimate.flags.writeable


@pytest.fixture(scope="module", params=["weibull", "lognormal", "piecewise_exponential"])
def fitted(request):
    name = request.param
    X, target, weight = sample("lognormal" if name == "lognormal" else "weibull")
    model = SurvivalGAM(likelihood=name, cuts=[0, .4, 1.2, 3] if name == "piecewise_exponential" else None,
                        formula="~x", optimize_smoothing=False).fit(X, target, sample_weight=weight)
    return model, X.iloc[:3]


@pytest.mark.parametrize("kind", ["hazard", "rmst"])
def test_joint_draw_endpoints_offsets_landmarks_and_persistence(fitted, kind, tmp_path):
    model, X = fitted
    p = len(model.gam_.fit_result().coef_full)
    offsets = np.full(len(X), .03) if model.likelihood_ == "piecewise_exponential" else np.tile([.03, -.02], (len(X), 1))
    z = np.random.default_rng(16).normal(size=(8, p))
    function = prepare_survival_hazard_interval if kind == "hazard" else prepare_survival_rmst_interval
    argument = {"times": [.2, .5, 1.5]} if kind == "hazard" else {"horizons": [0, .5, 1.5]}
    kwargs = {**argument, "offset": offsets, "conditional_on": [0, .5, 1.], "standard_draws": z}
    before = model.predict_link(X, offset=offsets)
    result = function(model, X, **kwargs)
    sampled = model.sample_survival_predictions(X, offset=offsets, standard_draws=z)
    if kind == "hazard":
        draws = np.stack([value.hazard(argument["times"]) for value in sampled])
    else:
        # Full rowwise integration gives the independent functional transport;
        # interval code instead subsets valid rows per shared restriction horizon.
        draws = np.full((len(sampled), len(X), 3), np.nan)
        for d, value in enumerate(sampled):
            for row, landmark in enumerate(kwargs["conditional_on"]):
                for j, horizon in enumerate(argument["horizons"]):
                    if horizon >= landmark:
                        draws[d, row, j] = value.take([row]).rmst(horizon, conditional_on=landmark)[0]
    valid = result.time[None, :] >= np.asarray(kwargs["conditional_on"])[:, None]
    expected = np.quantile(draws[:, valid], [.025, .975], axis=0)
    assert_allclose(result.lower[valid], expected[0], atol=1e-12)
    assert_allclose(result.upper[valid], expected[1], atol=1e-12)
    simultaneous = function(model, X, interval="simultaneous", **kwargs)
    inside = (draws[:, valid] >= simultaneous.lower[valid]-1e-12) & (draws[:, valid] <= simultaneous.upper[valid]+1e-12)
    assert np.all(inside)
    if kind == "rmst":
        width = result.time[None, :]-np.asarray(kwargs["conditional_on"])[:, None]
        assert np.all(simultaneous.lower[valid] >= 0)
        assert np.all(simultaneous.upper[valid] <= width[valid])
        assert simultaneous.lower[1, 1] == simultaneous.upper[1, 1] == 0
    replay = SurvivalGAM.load_model(model.save_model(tmp_path/"functional.nampy"))
    restored = function(replay, X, **kwargs)
    assert_array_equal(result.lower, restored.lower)
    assert_array_equal(result.upper, restored.upper)
    zero_kwargs = {**kwargs, "standard_draws": np.zeros((2, p))}
    endpoint = function(model, X, interval="simultaneous", **zero_kwargs)
    assert endpoint.critical_value == 0
    assert_allclose(endpoint.lower, endpoint.estimate, atol=1e-14, equal_nan=True)
    assert_array_equal(before, model.predict_link(X, offset=offsets))


def test_subject_selection_preserves_order_laws_and_support(fitted):
    model, X = fitted
    value = model.predict_distribution(X)
    selected = value.take([2, 0, 2])
    assert selected.backend == value.backend and selected.support == value.support
    assert_array_equal(selected.survival([0, 1, 2]), value.survival([0, 1, 2])[[2, 0, 2]])
    if selected.distribution is not None:
        assert all(not parameter.flags.writeable for parameter in selected.distribution.parameters.values())
    with pytest.raises(ValueError, match="one-dimensional"):
        value.take([[0, 1]])


def test_cut_conventions_zero_hazard_and_zero_remaining_lifetime():
    class Adapter:
        def predict_distribution(self, X, offset=None):
            return SurvivalPrediction(np.array([[0., 2.], [0., 0.]]), [0, 1, 3])
        def sample_survival_predictions(self, X, **kwargs):
            return [self.predict_distribution(X)]*2

    hazard = prepare_survival_hazard_interval(Adapter(), None, times=[0, 1, 2], interval="simultaneous")
    assert_array_equal(hazard.lower, [[0, 0, 2], [0, 0, 0]])
    assert_allclose(hazard.upper, hazard.lower)
    rmst = prepare_survival_rmst_interval(Adapter(), None, horizons=[0, 1, 2], conditional_on=[0, 1], interval="simultaneous")
    assert rmst.lower[0, 0] == 0 and rmst.lower[1, 1] == 0
    assert rmst.lower[1, 2] == rmst.upper[1, 2] == 1


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_smoothing_corrected_covariance_for_functionals(name):
    X, target = smooth_sample(name)
    model = SurvivalGAM(likelihood=name, formula=formulas(name)).fit(X, target)
    for function, argument in [(prepare_survival_hazard_interval, {"times": [.5, 1.]}),
                               (prepare_survival_rmst_interval, {"horizons": [.5, 1.]})]:
        result = function(model, X.iloc[:2], covariance="unconditional", n_draws=4, seed=8,
                          interval="simultaneous", **argument)
        assert result.uncertainty_kind == "coefficient_smoothing_corrected"
        assert np.isfinite(result.lower).all()


def test_unsupported_and_failed_routes_are_explicit(fitted, monkeypatch):
    model, X = fitted
    result = model.gam_.gam_result_
    core = result.fit_core_solution
    monkeypatch.setattr(model.gam_, "gam_result_", replace(result,
        fit_core_solution=replace(core, fit_result=replace(core.fit_result, cov_unconditional=None))))
    for function, argument in [(prepare_survival_hazard_interval, {"times": [.5]}),
                               (prepare_survival_rmst_interval, {"horizons": [.5]})]:
        with pytest.raises(ValueError, match="unavailable"):
            function(model, X, covariance="unconditional", **argument)
        with pytest.raises(NotImplementedError, match="covariance"):
            function(model, X, covariance="freq", **argument)
        with pytest.raises(ValueError, match="valid grid"):
            function(model, X, conditional_on=1., interval="simultaneous", n_draws=2, **argument)
    if model.likelihood_ != "piecewise_exponential":
        with pytest.raises(NotImplementedError, match="zero"):
            prepare_survival_hazard_interval(model, X, times=[0, 1])
    else:
        with pytest.raises(NotImplementedError, match="tail"):
            prepare_survival_rmst_interval(model, X, horizons=[4])
    def fail(*a, **kw):
        raise FloatingPointError("injected integration failure")
    monkeypatch.setattr(SurvivalPrediction, "rmst", fail)
    with pytest.raises(FloatingPointError, match="integration failure"):
        prepare_survival_rmst_interval(model, X, horizons=[1], n_draws=2)


def test_single_neural_functional_coefficient_routes_remain_unsupported(tmp_path):
    X, target, _ = sample("weibull")
    model = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood="weibull"),
        training=GradientTrainingConfig(max_epochs=1, batch_size=128, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})).fit(X, target)
    with pytest.raises(NotImplementedError, match="Neural survival coefficient"):
        prepare_survival_hazard_interval(model, X, times=[.5, 1])
    with pytest.raises(NotImplementedError, match="Neural survival coefficient"):
        prepare_survival_rmst_interval(model, X, horizons=[.5, 1], interval="simultaneous")


@pytest.mark.parametrize("interval", ["pointwise", "simultaneous"])
def test_rmst_bounds_remain_ordered_at_the_quadrature_upper_boundary(interval, monkeypatch):
    class Adapter:
        def predict_distribution(self, X, offset=None):
            return SurvivalPrediction(distribution=parametric_distribution("lognormal", [[1000., 1.]]))
        def sample_survival_predictions(self, X, **kwargs):
            return [self.predict_distribution(X)]*2

    assert_allclose(Adapter().predict_distribution(None).rmst(3), [3], atol=1e-14)
    # A one-ULP quadrature overshoot must not invert support-intersected bounds.
    # The fitted estimate is deliberately retained at its integration precision.
    def rounded_integral(self, horizon, *, conditional_on=None):
        width = horizon-self._landmarks(conditional_on, horizon)
        return np.where(width > 0, np.nextafter(width, np.inf), 0)
    monkeypatch.setattr(SurvivalPrediction, "rmst", rounded_integral)
    result = prepare_survival_rmst_interval(Adapter(), None, horizons=[0, 3], interval=interval)
    assert_array_equal(result.lower, [[0, 3]])
    assert_array_equal(result.upper, [[0, 3]])
    assert_allclose(result.estimate, [[0, 3]], atol=1e-14)
    assert result.estimate[0, 1] == np.nextafter(3., np.inf)
