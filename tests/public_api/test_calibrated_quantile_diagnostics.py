"""Calibrated inference declares its target and reports each optimization stage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from nampy import CalibratedQuantileGAM, QuantileGAM


@pytest.mark.parametrize("automatic", [False, True])
def test_fit_evidence_and_quantile_capabilities(automatic):
    data = pd.read_csv(Path(__file__).parents[1] / "reference_fixtures/qgam_calfast_auto_nonlinear_input.csv")
    X = data[["x", "z"]]
    estimator = CalibratedQuantileGAM(
        formula="~s(x,k=4)+z", level=0.3, bandwidth=0.2,
        smoothing_params=[0.02], optimize_smoothing=automatic,
        log_scale_bounds=(-4, 1),
    )
    assert estimator.capabilities()["uncertainty.quantile_function"].state == "conditional"
    model = estimator.fit(X, data.y)
    diagnostics = model.fit_diagnostics_
    coefficient = diagnostics["coefficients"]
    calibration = diagnostics["calibration"]
    assert coefficient["converged"] is True
    assert calibration["converged"] is True
    assert calibration["iterations"] > 0
    assert calibration["function_evaluations"] > 0
    assert calibration["optimality"] is None  # bounded solver has no score residual
    matrix = model.design_.matrix
    observation_score = (
        (model.level - 1 + expit((data.y.to_numpy() - model.predict(X)) / model.bandwidth_))
        / model.loss_scale_
    )
    penalty_score = model.penalty_ @ model.coef_
    score = penalty_score - matrix.T @ observation_score
    # The solver scales the score after subtraction; this recomputation scales
    # each term first. Near convergence their cancellation exposes roundoff.
    roundoff = 32 * np.finfo(float).eps * np.max(
        np.abs(penalty_score) + np.abs(matrix).T @ np.abs(observation_score)
    )
    assert_allclose(coefficient["optimality"], np.max(np.abs(score)), rtol=1e-5, atol=roundoff)
    assert_allclose(model.fit_result_.optimality, coefficient["optimality"])
    if automatic:
        core = model.gam_.fit_result().core
        assert coefficient["iterations"] == core.iter
        assert coefficient["iterations"] > 0
        assert diagnostics["smoothing"]["converged"] == model.gam_._optim_result.success
        assert diagnostics["smoothing"]["iterations"] == model.gam_._optim_result.nit
    else:
        assert coefficient["iterations"] is None  # hybr reports evaluations, not iterations
        assert coefficient["function_evaluations"] > 0
        assert diagnostics["smoothing"] is None
    capabilities = model.capabilities()
    assert capabilities["uncertainty.quantile_function"].state == "supported"
    assert capabilities["uncertainty.coefficients"].state == "supported"
    assert capabilities["predict_distribution"].state == "unsupported"
    interval = model.predict_interval(X.iloc[:3])
    expected = ("approximate_pointwise_conditional_on_selected_smoothing" if automatic
                else "approximate_pointwise_fixed_smoothing")
    assert interval.coverage == expected
    ordinary = QuantileGAM(formula="~x")
    assert ordinary.capabilities()["uncertainty.coefficients"].state == "unsupported"
    assert ordinary.capabilities()["uncertainty.quantile_function"].state == "unsupported"


def test_requested_automatic_route_without_free_penalties_reports_no_smoothing_stage():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    response = 0.3 + X.x.to_numpy() + 0.4 * np.sin(np.arange(len(X)) * 1.7)
    model = CalibratedQuantileGAM(
        formula="~x", bandwidth=0.2, optimize_smoothing=True,
    ).fit(X, response)
    assert model.gam_._optim_result is None
    assert model.fit_diagnostics_["smoothing"] is None
    assert model.fit_diagnostics_["coefficients"]["converged"] is True
    assert model.fit_diagnostics_["calibration"]["converged"] is True
    assert model.predict_interval(X.iloc[:2]).coverage == "approximate_pointwise_fixed_smoothing"
