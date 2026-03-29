import numpy as np
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _make_binomial_data,
    _fit_nampy_model_fixed_sp,
    _make_gamma_data,
    _make_negbin_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)
from nampy.gam.smoothing_selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)


pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


@pytest.mark.parametrize(
    ("family", "data_factory", "formula"),
    [
        ("poisson", lambda: _make_poisson_data(seed=53, n=220), 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ("poisson", lambda: _make_poisson_data(seed=57, n=220), 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ("binomial", lambda: _make_binomial_data(seed=67, n=220), 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ({"name": "negbin", "theta": 1.0}, lambda: _make_negbin_data(seed=61, n=240, theta=1.0), 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ({"name": "negbin", "theta": 1.0}, lambda: _make_negbin_data(seed=63, n=240, theta=1.0), 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
    ],
)
def test_known_scale_tensor_fixed_sp_reml_derivatives_match_mgcv(family, data_factory, formula):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "REML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    log_sp = np.log(sp)

    actual = float(criterion_value(gam, gam.y_, log_sp, method="reml"))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=1e-10,
        rtol=1e-10,
    )

    grad = np.asarray(criterion_gradient(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=2e-4,
        rtol=2e-4,
    )

    hess = np.asarray(criterion_hessian(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize(
    ("family", "data", "formula"),
    [
        ("poisson", _make_poisson_data(seed=71, n=220), 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ("binomial", _make_binomial_data(seed=73, n=220), 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ("gamma", _make_gamma_data(seed=101, n=220), 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        ({"name": "negbin", "theta": 1.0}, _make_negbin_data(seed=79, n=240, theta=1.0), 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
    ],
)
def test_t2_fixed_sp_reml_derivatives_match_mgcv_exactly(
    family,
    data,
    formula,
):
    expected = _run_mgcv_snapshot(data, formula, family, "REML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    log_sp = np.log(sp)

    actual = float(criterion_value(gam, gam.y_, log_sp, method="reml"))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=1e-10,
        rtol=1e-10,
    )

    grad = np.asarray(criterion_gradient(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=2e-7,
        rtol=2e-7,
    )

    hess = np.asarray(criterion_hessian(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=5e-6,
        rtol=5e-6,
    )


def test_gamma_tensor_te_fixed_sp_reml_matches_mgcv():
    data = _make_gamma_data(seed=97, n=220)
    formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "gamma", sp)
    log_sp = np.log(sp)

    actual = float(criterion_value(gam, gam.y_, log_sp, method="reml"))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=1e-10,
        rtol=1e-10,
    )

    grad = np.asarray(criterion_gradient(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=2e-4,
        rtol=2e-4,
    )

    hess = np.asarray(criterion_hessian(gam, gam.y_, log_sp, method="reml"), dtype=np.float64)
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=2e-3,
        rtol=2e-3,
    )
