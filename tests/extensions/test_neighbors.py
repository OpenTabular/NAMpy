"""Independent upstream outputs and statistical invariants for extensions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from nampy import GAMLSS, ExpectileGAM, GAMRegressor, QuantileGAM
from nampy.gam.inference.functionals import (
    coefficient_draws,
    derivative_interval,
    joint_edf,
    linear_functional_interval,
)
from nampy.gam.losses.objectives import fit_expectile, fit_quantile
from nampy.gam.structured.lags import (
    cross_basis,
    cross_basis_penalty,
    cumulative_lag_matrix,
    exposure_history,
)
from nampy.survival import piecewise_exponential_data, survival_from_hazard

FIXTURE = Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"


@pytest.fixture(scope="module")
def reference():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("index", [0, 1, 2])
def test_expectiles_match_unmodified_expectreg_solver(reference, index):
    d = reference["inputs"]
    expected = reference["upstream"]["expectiles"][index]
    fit = fit_expectile(
        d["design"],
        d["response"],
        level=expected["level"],
        penalty=d["penalty"],
        weights=d["weights"],
    )
    assert fit.converged
    assert_allclose(fit.coefficients, expected["coefficients"], atol=2e-10)
    assert fit.optimality < 1e-8


@pytest.mark.parametrize("index", [0, 1, 2])
def test_smooth_quantiles_match_qgam_elf_source(reference, index):
    d = reference["inputs"]
    expected = reference["upstream"]["elf"][index]
    fit = fit_quantile(
        d["design"],
        d["response"],
        level=expected["level"],
        penalty=d["penalty"],
        weights=d["weights"],
        bandwidth=d["bandwidth"],
    )
    assert fit.converged
    assert_allclose(fit.coefficients, expected["coefficients"], atol=2e-6, rtol=0)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_exact_quantiles_match_quantreg_and_penalized_quadprog(reference, index):
    d = reference["inputs"]
    expected = reference["upstream"]["quantiles"][index]
    fit = fit_quantile(
        d["design"], d["response"], level=expected["level"], weights=d["weights"]
    )
    assert fit.converged
    assert_allclose(fit.coefficients, expected["coefficients"], atol=2e-7, rtol=0)
    fit = fit_quantile(
        d["design"],
        d["response"],
        level=expected["level"],
        weights=d["weights"],
        penalty=d["penalty"],
    )
    assert fit.converged
    assert_allclose(
        fit.coefficients,
        reference["upstream"]["penalized_quantiles"][index]["coefficients"],
        atol=2e-5,
        rtol=0,
    )


def test_half_expectile_is_penalized_least_squares(reference):
    d = reference["inputs"]
    X, y, S, w = (
        np.asarray(d[k]) for k in ["design", "response", "penalty", "weights"]
    )
    fit = fit_expectile(X, y, penalty=S, weights=w)
    expected = np.linalg.solve(X.T @ (w[:, None] * X) + S, X.T @ (w * y))
    assert_allclose(fit.coefficients, expected, atol=1e-12)


def test_dlnm_crossbasis_order_and_group_boundaries(reference):
    d = reference["inputs"]
    history = np.asarray(d["history"])
    lag = np.column_stack([np.ones(4), np.arange(4)])
    actual = cross_basis(
        history, exposure_basis=lambda x: np.column_stack([x, x**2]), lag_basis=lag
    )
    assert_allclose(actual, reference["upstream"]["cross_basis"], atol=1e-13)
    h = exposure_history(d["series"], max_lag=2, groups=d["groups"])
    actual = cross_basis(h, lag_basis=np.column_stack([np.ones(3), np.arange(3)]))
    assert_allclose(
        actual,
        np.asarray(reference["upstream"]["series_cross_basis"], dtype=float),
        atol=1e-13,
    )
    h[2, 0] = np.nan
    assert np.isnan(cross_basis(h)[2]).all()


def _mock_model(d):
    result = SimpleNamespace(coef_full=np.asarray(d["beta"]))
    core = SimpleNamespace(fit_result=result)
    return SimpleNamespace(
        _fitted=True,
        gam_result_=SimpleNamespace(fit_core_solution=core),
        _select_cov=lambda covariance: np.asarray(d["covariance"]),
    )


def test_joint_simultaneous_bands_match_actual_gratia_kernel(reference):
    d = reference["inputs"]
    m = _mock_model(d)
    actual = linear_functional_interval(
        m, d["functional"], standard_draws=d["standard_draws"]
    )
    expected = reference["upstream"]
    assert_allclose(actual.estimate, expected["band_estimate"], atol=1e-14)
    assert_allclose(actual.standard_error, expected["band_se"], atol=1e-14)
    assert_allclose(actual.critical_value, expected["band"]["critical"], atol=1e-13)
    assert_allclose(actual.lower, expected["band"]["lower"], atol=1e-13)
    assert_allclose(actual.upper, expected["band"]["upper"], atol=1e-13)
    draws = coefficient_draws(m, standard_draws=d["standard_draws"])
    assert_allclose(draws, np.asarray(d["deviations"]) + d["beta"], atol=1e-14)
    zero = linear_functional_interval(m, np.zeros((2, 3)), n_draws=10)
    assert_allclose(zero.lower, 0)
    assert not zero.significant.any()


def test_unconditional_covariance_requires_the_actual_correction(reference):
    d = reference["inputs"]
    model = _mock_model(d)
    fit = model.gam_result_.fit_core_solution.fit_result
    fit.cov_unconditional = None
    with pytest.raises(ValueError, match="unavailable"):
        coefficient_draws(model, covariance="unconditional", n_draws=2)
    fit.cov_unconditional = np.asarray(d["covariance"]) * 2
    draws = coefficient_draws(
        model, covariance="unconditional", standard_draws=d["standard_draws"]
    )
    assert_allclose(
        draws,
        np.asarray(d["beta"]) + np.sqrt(2) * np.asarray(d["deviations"]),
        atol=1e-13,
    )


def test_survival_split_matches_upstream_survival_kernel(reference):
    d = reference["inputs"]
    frame = pd.DataFrame(d["survival"])
    actual = piecewise_exponential_data(
        frame, time="time", event="event", entry="entry", id="subject", cuts=d["cuts"]
    )
    expected = pd.DataFrame(reference["upstream"]["survival_split"])
    assert_allclose(actual["ped_event"], expected["event"])
    assert_allclose(actual["tstart"], expected["tstart"])
    assert_allclose(actual["exposure"], expected["exposure"])
    assert_allclose(actual["offset"], expected["offset"])
    assert_allclose(
        actual.groupby("ped_id")["exposure"].sum(), frame["time"] - frame["entry"]
    )
    assert actual["ped_event"].sum() == frame["event"].sum()
    # The event at cut 1 is counted once, in (0,1].
    assert actual.loc[actual.ped_id == 1, "interval"].tolist() == [0]


def test_survival_integration_partial_intervals_and_monotonicity():
    result = survival_from_hazard(
        [[0.2, 0.4], [0, 0.1]], [0, 1, 3], times=[0, 0.5, 1, 2, 3]
    )
    assert_allclose(result["cumulative_hazard"][0], [0, 0.1, 0.2, 0.6, 1])
    assert_allclose(result["survival"], np.exp(-result["cumulative_hazard"]))
    assert (np.diff(result["survival"], axis=1) <= 0).all()


def test_lag_cumulative_contrasts_and_tensor_penalty():
    lag = np.column_stack([np.ones(3), np.arange(3)])
    L = cumulative_lag_matrix([1, 2], 1, n_lags=3, lag_basis=lag)
    assert_allclose(L, [[0, 0], [3, 3]])
    P = cross_basis_penalty(
        np.diag([1, 2]), np.diag([3, 4]), exposure_weight=2, lag_weight=0.5
    )
    assert_allclose(P, np.diag([3.5, 4, 5.5, 6]))


def test_joint_edf_limits_reparameterization_and_cross_coupling():
    information = np.array([[4.0, 1.3], [1.3, 2.0]])
    S = np.diag([0.2, 3])
    edf, total, edf1 = joint_edf(information, S)
    F = np.linalg.solve(information + S, information)
    assert_allclose(edf, np.diag(F))
    assert_allclose(edf1, np.diag(2 * F - F @ F))
    assert total != pytest.approx(
        np.sum(np.diag(information) / (np.diag(information) + np.diag(S)))
    )
    A = np.array([[1.0, 2], [0.0, 0.7]])
    assert joint_edf(A.T @ information @ A, A.T @ S @ A)[1] == pytest.approx(total)
    assert joint_edf(information, np.zeros((2, 2)))[1] == pytest.approx(2)
    assert joint_edf(information, 1e10 * np.eye(2))[1] < 1e-8
    assert joint_edf(information, np.diag([0, 1e10]))[1] == pytest.approx(1, abs=1e-8)


@pytest.mark.parametrize("cls", [ExpectileGAM, QuantileGAM])
def test_estimators_formula_offsets_components_and_sklearn_clone(cls):
    x = np.linspace(-1, 1, 50)
    data = pd.DataFrame(
        {"x": x, "y": np.sin(3 * x) + 0.1 * np.cos(np.arange(50)), "off": 0.2 * x}
    )
    options = {"bandwidth": 0.1} if cls is QuantileGAM else {}
    model = clone(
        cls(
            formula="y ~ s(x,k=6) + offset(off)",
            level=0.7,
            smoothing_params=[0.3],
            **options,
        )
    ).fit(data)
    assert model.fit_result_.converged
    prediction = model.predict(data)
    components = model.predict_components(data)
    reconstructed = (
        components.intercept
        + components.offset
        + sum(components.terms.values())
    )
    assert_allclose(prediction[:, None], reconstructed)
    assert prediction.shape == (50,)
    changed = data.assign(off=data.off + 2)
    assert_allclose(model.predict(changed) - prediction, 2)
    assert model.score(data, data.y) <= 0


def test_expectile_midpoint_matches_existing_gaussian_gam():
    x = np.linspace(0, 1, 60)
    data = pd.DataFrame({"x": x, "y": np.sin(5 * x) + 0.15 * np.cos(np.arange(60))})
    a = ExpectileGAM(formula="y~s(x,k=6)", smoothing_params=[0.2]).fit(data)
    b = GAMRegressor(
        formula="y~s(x,k=6)", optimize_smoothing=False, smoothing_params=[0.2]
    ).fit(data)
    assert_allclose(a.predict(data), b.predict(data), atol=1e-10)


def test_real_model_derivatives_and_joint_gaulss_edf():
    from nampy.gam.model_state import _fit_result

    x = np.linspace(-1, 1, 90)
    data = pd.DataFrame(
        {"x": x, "y": 0.3 + 0.5 * x + np.exp(0.2 * x) * np.sin(np.arange(90) * 1.7)}
    )
    model = GAMRegressor(formula="y~x", optimize_smoothing=False).fit(data)
    band = derivative_interval(model, data, "x", interval="pointwise")
    assert_allclose(
        band.estimate,
        model.predict(data.assign(x=x + 1)) - model.predict(data),
        atol=1e-10,
    )
    joint = GAMLSS(
        formula={"mu": "y~s(x,k=5)", "sigma": "~s(x,k=4)"}, optimize_smoothing=False
    ).fit(data)
    fit = _fit_result(joint.gam_)
    # Existing general-family postprocessing retains the joint information
    # matrix; no per-predictor block-diagonal substitution is acceptable.
    state = joint.gam_.gam_result_.fit_core_solution.fit_state
    assert fit.edf > 0
    assert np.isfinite(fit.edf)
    assert_allclose(np.sum(fit.edf_by_term) + 2, fit.edf, atol=1e-7)
    assert state is not None


@pytest.mark.parametrize("solver", [fit_expectile, fit_quantile])
@pytest.mark.parametrize(
    "options",
    [
        {"level": 0},
        {"level": 1},
        {"weights": [0, 0]},
        {"weights": [1, -1]},
        {"penalty": [[-1]]},
        {"max_iter": 0},
        {"tol": 0},
    ],
)
def test_invalid_objective_inputs_rejected(solver, options):
    with pytest.raises(ValueError):
        solver([[1], [1]], [1, 2], **options)


def test_invalid_applied_inputs_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        exposure_history([1, 2, 3], max_lag=1, groups=[1, 2, 1])
    with pytest.raises(ValueError):
        survival_from_hazard([[-1]], [0, 1])
    with pytest.raises(ValueError):
        joint_edf(np.eye(2), -np.eye(2))
