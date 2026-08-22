"""SCAM GCV/UBRE value and exact-gradient parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import criterion_gradient, criterion_value
from tests.scam.scam_reference_utils import (
    run_scam_fixed_sp_fit,
    run_scam_selected_sp_fit,
)


@pytest.mark.parametrize(
    "python_family,r_family,method",
    [
        pytest.param("gaussian", "gaussian", "gcv", id="gaussian-gcv"),
        pytest.param("poisson", "poisson", "ubre", id="poisson-ubre"),
        pytest.param(("gamma", "log"), "Gamma(link='log')", "gcv", id="gamma-gcv"),
    ],
)
def test_shape_smoothing_criterion_and_gradient_match_scam(
    python_family, r_family, method
):
    rng = np.random.default_rng(2512)
    x = np.sort(rng.uniform(-1.7, 2.5, size=180))
    signal = -0.7 + 1.8 / (1.0 + np.exp(-1.4 * x))
    if python_family == "gaussian":
        y = signal + rng.normal(scale=0.12, size=x.size)
    elif python_family == "poisson":
        y = rng.poisson(np.exp(signal)).astype(np.float64)
    else:
        mean = np.exp(signal)
        y = rng.gamma(7.0, mean / 7.0).astype(np.float64)
    data = pd.DataFrame({"y": y, "x": x})
    formula = "y ~ s(x, bs='mpi', k=8, m=2)"
    start = np.array([0.12, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2])
    smoothing_parameter = 0.63
    log_sp = np.array([np.log(smoothing_parameter)])
    model = GAM(
        formula=formula,
        family=python_family,
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[smoothing_parameter],
        start=start,
    ).fit(data=data)

    actual_value = criterion_value(model, y, log_sp, method=method)
    actual_gradient = criterion_gradient(model, y, log_sp, method=method)
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family=r_family,
        sp=[smoothing_parameter],
        start=start,
    )
    score_key = "gcv_score" if method == "gcv" else "ubre_score"
    np.testing.assert_allclose(
        actual_value, expected[score_key], rtol=4e-8, atol=4e-10
    )

    step = 1e-5
    expected_minus = run_scam_fixed_sp_fit(
        data,
        formula,
        family=r_family,
        sp=[smoothing_parameter * np.exp(-step)],
        start=start,
    )[score_key]
    expected_plus = run_scam_fixed_sp_fit(
        data,
        formula,
        family=r_family,
        sp=[smoothing_parameter * np.exp(step)],
        start=start,
    )[score_key]
    expected_gradient = (expected_plus - expected_minus) / (2.0 * step)
    np.testing.assert_allclose(
        actual_gradient,
        np.array([expected_gradient]),
        rtol=2e-5,
        atol=2e-7,
    )


@pytest.mark.parametrize(
    "python_family,r_family,method",
    [
        pytest.param("gaussian", "gaussian", "gcv", id="gaussian-gcv"),
        pytest.param("poisson", "poisson", "ubre", id="poisson-ubre"),
    ],
)
def test_shape_bfgs_smoothing_selection_matches_scam(
    python_family, r_family, method
):
    rng = np.random.default_rng(2513)
    x = np.sort(rng.uniform(-1.7, 2.5, size=180))
    signal = -0.7 + 1.8 / (1.0 + np.exp(-1.4 * x))
    if python_family == "gaussian":
        y = signal + rng.normal(scale=0.12, size=x.size)
    else:
        y = rng.poisson(np.exp(signal)).astype(np.float64)
    data = pd.DataFrame({"y": y, "x": x})
    formula = "y ~ s(x, bs='mpi', k=8, m=2)"
    start = np.array([0.12, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2])

    expected = run_scam_selected_sp_fit(
        data,
        formula,
        family=r_family,
        start=start,
    )
    model = GAM(
        formula=formula,
        family=python_family,
        smoothing_method=method,
        smoothing_optimizer="bfgs",
        optimize_smoothing=True,
        start=start,
    ).fit(data=data)
    actual = model.fit_result()

    np.testing.assert_allclose(model.smoothing_params, expected["sp"], rtol=2e-5)
    np.testing.assert_allclose(
        actual.coef_optimization, expected["coefficients"], rtol=2e-5
    )
    np.testing.assert_allclose(actual.mu, expected["mu"], rtol=2e-5)
    np.testing.assert_allclose(model.smoothing_score_, expected["score"], rtol=2e-5)


def test_shape_smoothing_selection_rejects_non_scam_outer_optimizer():
    x = np.linspace(-1.0, 1.0, 40)
    data = pd.DataFrame({"x": x, "y": np.exp(x)})
    model = GAM(
        formula="y ~ s(x, bs='mpi', k=7)",
        family="gaussian",
        smoothing_method="gcv",
        smoothing_optimizer="optim",
        optimize_smoothing=True,
    )
    with pytest.raises(NotImplementedError, match="bfgs_gcv.ubre"):
        model.fit(data=data)
