from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_negbin_data,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


def test_strict_t2_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_strict_factor_by_link_parity():
    rng = np.random.default_rng(31)
    n = 80
    x = rng.normal(size=n)
    fac = np.array(["p", "q"] * (n // 2), dtype=object)
    y = np.sin(x) + 0.4 * (fac == "q").astype(float) + rng.normal(0, 0.15, n)
    data = pd.DataFrame({"y": y, "x": x, "fac": fac})
    formula = 'y ~ s(x, by=fac, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=5e-10,
        rtol=0.0,
    )


def test_strict_poisson_reml_residual_parity():
    case = get_parity_case("poisson_cr_uni_reml")
    data = make_parity_case_data(case.case_id)
    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    model = _fit_nampy_model_fixed_sp(data, case.formula, case.family, sp)
    actual = model.parity_snapshot(X=data, include_covariances=True)
    a_res = actual["parity"]["diagnostics"]["residuals"]
    e_res = expected["parity"]["diagnostics"]["residuals"]

    for key in ("response", "working", "pearson", "scaled_pearson", "deviance"):
        np.testing.assert_allclose(
            np.asarray(a_res[key], dtype=np.float64),
            np.asarray(e_res[key], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )


def test_strict_binomial_reml_residual_parity():
    case = get_parity_case("binomial_cr_uni_reml")
    data = make_parity_case_data(case.case_id)
    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    model = _fit_nampy_model_fixed_sp(data, case.formula, case.family, sp)
    actual = model.parity_snapshot(X=data, include_covariances=True)
    a_res = actual["parity"]["diagnostics"]["residuals"]
    e_res = expected["parity"]["diagnostics"]["residuals"]

    for key in ("response", "working", "pearson", "scaled_pearson", "deviance"):
        np.testing.assert_allclose(
            np.asarray(a_res[key], dtype=np.float64),
            np.asarray(e_res[key], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )


def test_tensor_te_ps_ps_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ te(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_tensor_ti_ps_ps_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ ti(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_tensor_t2_ps_ps_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ t2(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3, 0.9])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_negbin_estimated_theta_reml_endpoint_gap_tracked():
    data = _make_negbin_data(seed=2024, n=240, theta=1.0)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

    model = _fit_nampy_model(data, formula, family, "REML")
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=2e-4,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        atol=1e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=2e-2,
        rtol=2e-2,
    )

    actual_log_sp = np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64)
    expected_log_sp = np.asarray(
        expected["fit"]["log_smoothing_params"], dtype=np.float64
    )
    assert np.all(np.isfinite(actual_log_sp))
    assert np.all(np.isfinite(expected_log_sp))
    np.testing.assert_allclose(
        actual_log_sp,
        expected_log_sp,
        atol=0.2,
        rtol=0.0,
    )

    endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
    assert endpoint is not None
    assert endpoint["joint_negbin_reml_outer"] is True
    assert endpoint["joint_negbin_postprocessed"] is True
    assert endpoint["joint_negbin_flat_ridge_stabilized"] is True
    assert endpoint["joint_log_theta"] is not None
    assert endpoint["family_theta"] == pytest.approx(
        actual["fit"]["family_theta"], abs=5e-5
    )
    assert endpoint["joint_negbin_optimizer_message"] is not None


def test_negbin_estimated_theta_reml_two_smooth_theta2_gap_tracked():
    data = _make_negbin_data(seed=341, n=240, theta=2.0)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

    model = _fit_nampy_model(data, formula, family, "REML")
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=2e-2,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        atol=4e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=1e-1,
        rtol=1e-1,
    )

    actual_log_sp = np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64)
    expected_log_sp = np.asarray(
        expected["fit"]["log_smoothing_params"], dtype=np.float64
    )
    assert np.all(np.isfinite(actual_log_sp))
    assert np.all(np.isfinite(expected_log_sp))
    np.testing.assert_allclose(actual_log_sp, expected_log_sp, atol=0.05, rtol=0.0)

    endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
    assert endpoint is not None
    assert endpoint["joint_negbin_reml_outer"] is True
    assert endpoint["joint_negbin_postprocessed"] is True
    assert endpoint["joint_negbin_flat_ridge_stabilized"] is True
    assert endpoint["joint_log_theta"] is not None
    assert endpoint["family_theta"] == pytest.approx(
        actual["fit"]["family_theta"], abs=5e-5
    )


def test_negbin_estimated_theta_reml_two_smooth_theta05_gap_tracked():
    data = _make_negbin_data(seed=340, n=240, theta=0.5)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 0.5, "estimate_theta": True}

    model = _fit_nampy_model(data, formula, family, "REML")
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=3e-3,
        rtol=3e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        atol=2e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=3e-2,
        rtol=3e-2,
    )

    actual_log_sp = np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64)
    expected_log_sp = np.asarray(
        expected["fit"]["log_smoothing_params"], dtype=np.float64
    )
    assert np.all(np.isfinite(actual_log_sp))
    assert np.all(np.isfinite(expected_log_sp))
    np.testing.assert_allclose(actual_log_sp, expected_log_sp, atol=0.35, rtol=0.0)

    endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
    assert endpoint is not None
    assert endpoint["joint_negbin_reml_outer"] is True
    assert endpoint["joint_negbin_postprocessed"] is True
    assert endpoint["joint_negbin_flat_ridge_stabilized"] is True
    assert endpoint["joint_log_theta"] is not None
    assert endpoint["family_theta"] == pytest.approx(actual["fit"]["family_theta"])
