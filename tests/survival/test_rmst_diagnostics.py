"""Shared residuals and RMST against independent formulas, including entry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import gamma, gammainc, ndtr

from nampy.survival import SurvivalPrediction, SurvivalTarget, survival_residuals
from nampy.survival.likelihood import parametric_distribution


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
@pytest.mark.parametrize("horizon", [0., .01, 1., 10.])
def test_parametric_rmst_against_analytic_partial_moments(name, horizon):
    parameters = np.array([[.6, .5], [1.2, 1.], [2., 3.]])
    prediction = SurvivalPrediction(distribution=parametric_distribution(name, parameters))
    if horizon == 0:
        expected = np.zeros(3)
    elif name == "weibull":
        scale, shape = parameters.T
        expected = scale*gamma(1+1/shape)*gammainc(1/shape, (horizon/scale)**shape)
    else:
        loc, sd = parameters.T
        z = (np.log(horizon)-loc)/sd
        expected = np.exp(loc+sd**2/2)*ndtr(z-sd)+horizon*ndtr(-z)
    assert_allclose(prediction.rmst(horizon), expected, atol=2e-9, rtol=2e-9)
    assert np.all(prediction.rmst(horizon) <= horizon)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_remaining_rmst_from_subject_landmarks(name):
    parameters = np.array([[.8, 1.5], [1.2, .7]])
    prediction = SurvivalPrediction(distribution=parametric_distribution(name, parameters))
    a = np.array([.2, 1.1])
    # Independent antiderivative difference on a moderate range, avoiding the
    # direct subtraction used neither by quadrature nor hazard increments.
    if name == "weibull":
        scale, shape = parameters.T
        def primitive(t):
            return scale*gamma(1+1/shape)*gammainc(1/shape, (t/scale)**shape)
    else:
        loc, sd = parameters.T
        def primitive(t):
            z = (np.log(t)-loc)/sd
            return np.exp(loc+sd**2/2)*ndtr(z-sd)+t*ndtr(-z)
    expected = (primitive(3)-primitive(a))/prediction.distribution.survival(a)
    assert_allclose(prediction.rmst(3, conditional_on=a), expected, rtol=2e-9, atol=2e-9)
    assert_array_equal(prediction.rmst(1, conditional_on=1), [0, 0])
    with pytest.raises(ValueError, match="landmark"):
        prediction.rmst(1, conditional_on=[.1, 2])


def test_piecewise_conditional_rmst_handles_zero_hazard_and_partial_intervals():
    rates = np.array([[0., .5, 1.], [.3, 0, .7]])
    prediction = SurvivalPrediction(rates, [0, 1, 2, 4])
    expected = [
        -np.expm1(-.5)/.5+np.exp(-.5)*-np.expm1(-1),
        -np.expm1(-.3*.5)/.3+np.exp(-.3*.5)*(1+-np.expm1(-.7)/.7),
    ]
    assert_allclose(prediction.rmst(3, conditional_on=[1, .5]), expected)
    assert_array_equal(prediction.rmst(0), [0, 0])
    with pytest.raises(NotImplementedError, match="tail"):
        prediction.rmst(5)


@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
def test_residuals_are_rowwise_and_entry_conditioned(name):
    target = SurvivalTarget([1., 2., 3.], [1, 0, 1], [0, .2, 1.], ["a", "b", "c"])
    if name == "piecewise_exponential":
        prediction = SurvivalPrediction(np.array([[.2, .4], [.3, .2], [.6, .8]]), [0, 1, 4])
    else:
        prediction = SurvivalPrediction(distribution=parametric_distribution(
            name, [[1, .8], [1.2, 1.1], [1.5, .6]]))
    residual = survival_residuals(prediction, target)
    grid = np.unique(np.r_[target.time, target.entry])
    curves = prediction.cumulative_hazard(grid)
    i = np.arange(3)
    expected = curves[i, np.searchsorted(grid, target.time)]-curves[i, np.searchsorted(grid, target.entry)]
    assert_allclose(residual.cox_snell, expected)
    assert_allclose(residual.martingale, target.event-expected)
    m = target.event-expected
    assert_allclose(residual.deviance,
                    np.sign(m)*np.sqrt(-2*(m+target.event*np.log(expected))))
    assert_array_equal(residual.event, target.event)
    assert_array_equal(residual.subject_id, target.subject_id)
    assert not residual.cox_snell.flags.writeable


def test_zero_hazard_residuals_keep_impossible_events_explicit():
    target = SurvivalTarget([1, 2], [0, 1])
    prediction = SurvivalPrediction(np.zeros((2, 1)), [0, 3])
    result = survival_residuals(prediction, target)
    assert_array_equal(result.cox_snell, [0, 0])
    assert_array_equal(result.martingale, [0, 1])
    assert result.deviance[0] == 0 and np.isposinf(result.deviance[1])


def test_quadrature_failures_are_explicit(monkeypatch):
    prediction = SurvivalPrediction(distribution=parametric_distribution("weibull", [[1., 2.]]))
    monkeypatch.setattr("nampy.survival.prediction.quad", lambda *a, **k: (0., 1., {}, "failed"))
    with pytest.raises(FloatingPointError, match="converge"):
        prediction.rmst(2)


def test_log_space_rmst_works_when_unrestricted_lognormal_mean_overflows():
    prediction = SurvivalPrediction(distribution=parametric_distribution("lognormal", [[1000., 1.]]))
    assert_allclose(prediction.rmst(3), [3])
    prediction = SurvivalPrediction(distribution=parametric_distribution("lognormal", [[-20., .5]]))
    # Conditional survival at 1 underflows, but remaining RMST is still finite.
    assert prediction.distribution.survival(1)[0] == 0
    result = prediction.rmst(1.1, conditional_on=1)[0]
    assert 0 < result < .1


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_large_horizon_does_not_hide_early_survival_mass(name):
    parameters = [[1., 1.]] if name == "weibull" else [[0., 1.]]
    prediction = SurvivalPrediction(distribution=parametric_distribution(name, parameters))
    assert_allclose(prediction.rmst(1e100), prediction.distribution.mean(), rtol=2e-9)
    if name == "weibull":
        assert_allclose(prediction.rmst(1e100, conditional_on=.5), [1], rtol=2e-9)
