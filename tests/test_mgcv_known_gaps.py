from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
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
        atol=1e-5,
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
        atol=1e-7,
        rtol=1e-7,
    )


def test_tensor_ti_ps_ps_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ ti(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )


def test_tensor_t2_ps_ps_fixed_sp_response_parity():
    data = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ t2(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3, 0.9])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )
