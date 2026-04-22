from __future__ import annotations

import numpy as np

from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_negbin_data,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)


def test_strict_t2_fixed_sp_response_parity():
    """Known-gap coverage verifying that strict t2 fixed sp response parity."""
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


def test_strict_poisson_reml_residual_parity():
    """Known-gap coverage verifying that strict poisson REML residual parity."""
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
            atol=2e-10,
            rtol=2e-10,
        )


def test_strict_binomial_reml_residual_parity():
    """Known-gap coverage verifying that strict binomial REML residual parity."""
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
            atol=5e-10,
            rtol=5e-10,
        )


def test_tensor_te_ps_ps_fixed_sp_response_parity():
    """Known-gap coverage verifying that tensor te ps ps fixed sp response parity."""
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


def test_tensor_te_ps_ps_margin_orders_fixed_sp_response_parity():
    """
    Known-gap coverage verifying that tensor te ps ps margin orders fixed sp response
    parity.
    """
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_tensor_ti_ps_ps_fixed_sp_response_parity():
    """Known-gap coverage verifying that tensor ti ps ps fixed sp response parity."""
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


def test_tensor_ti_ps_ps_margin_orders_fixed_sp_response_parity():
    """
    Known-gap coverage verifying that tensor ti ps ps margin orders fixed sp response
    parity.
    """
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ ti(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_tensor_t2_ps_ps_fixed_sp_response_parity():
    """Known-gap coverage verifying that tensor t2 ps ps fixed sp response parity."""
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


def test_tensor_t2_ps_ps_margin_orders_fixed_sp_response_parity():
    """
    Known-gap coverage verifying that tensor t2 ps ps margin orders fixed sp response
    parity.
    """
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = (
        'y ~ t2(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], ' "sp=[0.7, 1.3, 0.9])"
    )
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_negbin_estimated_theta_reml_endpoint_matches_mgcv():
    """
    Known-gap coverage verifying that negative-binomial estimated theta REML endpoint
    matches mgcv.
    """
    data = _make_negbin_data(seed=2024, n=240, theta=1.0)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=1e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )


def test_negbin_estimated_theta_reml_two_smooth_theta2_matches_mgcv():
    """
    Known-gap coverage verifying that negative-binomial estimated theta REML two smooth
    theta2 matches mgcv.
    """
    data = _make_negbin_data(seed=341, n=240, theta=2.0)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=5e-3,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )


def test_negbin_estimated_theta_reml_two_smooth_theta05_matches_mgcv():
    """
    Known-gap coverage verifying that negative-binomial estimated theta REML two smooth
    theta05 matches mgcv.
    """
    data = _make_negbin_data(seed=340, n=240, theta=0.5)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 0.5, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]

    assert bool(endpoint["joint_negbin_reml_outer"]) is True
    assert bool(endpoint["joint_negbin_efs_outer"]) is True
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=2e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=2e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=3e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=2e-6,
        rtol=0.0,
    )
