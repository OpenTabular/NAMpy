from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.smoothing_selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_snapshot,
)


def _gaulss_data(seed=11, n=140):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    mu = 0.3 + np.sin(np.pi * x)
    sigma = np.exp(-0.35 + 0.25 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _gaulss_by_data(seed=21, n=160):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    z = rng.uniform(0.5, 1.5, size=n)
    mu = 0.3 + np.sin(np.pi * x) * z
    sigma = np.exp(-0.35 + 0.25 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _gaulss_tensor_data(seed=22, n=160):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = 0.25 + np.sin(np.pi * x0) + 0.3 * x1**2
    sigma = np.exp(-0.35 + 0.2 * x0 - 0.1 * x1)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _gammals_data(n=100, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.4 + 0.3 * x)
    phi = np.exp(-0.5)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x": x})


def _gammals_by_data(seed=23, n=140):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.25, 1.25, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    mu = np.exp(0.35 + 0.3 * np.sin(np.pi * x) * z)
    phi = np.exp(-0.5)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _gammals_tensor_data(seed=24, n=160):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = np.exp(0.35 + np.sin(np.pi * x0) + 0.25 * x1**2)
    phi = np.exp(-0.45)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _sample_ziplss_response(rng, gamma, eta):
    from scipy.stats import poisson

    lam = np.exp(gamma)
    p = 1.0 - np.exp(-np.exp(eta))
    y = np.zeros_like(lam)
    ind = rng.uniform(size=lam.shape[0]) < p
    u = rng.uniform(size=int(ind.sum()))
    u = poisson.cdf(0, lam[ind]) + u * (1.0 - poisson.cdf(0, lam[ind]))
    y[ind] = poisson.ppf(np.minimum(u, 1.0 - 1e-12), lam[ind])
    return y


def _ziplss_data(n=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    gamma = 0.2 + 0.4 * x
    eta = np.full(n, -0.3)
    y = _sample_ziplss_response(rng, gamma, eta)
    return pd.DataFrame({"y": y, "x": x})


def _ziplss_by_data(seed=25, n=160):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.25, 1.25, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    gamma = 0.15 + 0.4 * np.sin(np.pi * x) * z
    eta = -0.35 + 0.2 * x
    y = _sample_ziplss_response(rng, gamma, eta)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _ziplss_tensor_data(seed=26, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    gamma = 0.15 + 0.45 * np.sin(np.pi * x0) + 0.2 * x1**2
    eta = -0.3 + 0.25 * x0 - 0.15 * x1
    y = _sample_ziplss_response(rng, gamma, eta)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _gevlss_data(n=90, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = 0.2 + 0.5 * x
    rho = np.full(n, -0.4)
    xi = np.full(n, 0.1)
    u = rng.uniform(size=n)
    y = mu + ((-np.log(u)) ** (-xi) - 1.0) * np.exp(rho) / xi
    return pd.DataFrame({"y": y, "x": x})


def _sample_gev_response(rng, mu, rho, xi):
    u = rng.uniform(size=np.asarray(mu).shape[0])
    return mu + ((-np.log(u)) ** (-xi) - 1.0) * np.exp(rho) / xi


def _gevlss_by_data(seed=27, n=140):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.25, 1.25, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    mu = 0.2 + 0.45 * np.sin(np.pi * x) * z
    rho = -0.35 + 0.1 * x
    xi = np.full(n, 0.1)
    y = _sample_gev_response(rng, mu, rho, xi)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _gevlss_tensor_data(seed=28, n=160):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = 0.25 + 0.4 * np.sin(np.pi * x0) + 0.2 * x1**2
    rho = -0.35 + 0.15 * x0 - 0.1 * x1
    xi = np.full(n, 0.1)
    y = _sample_gev_response(rng, mu, rho, xi)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _gevlss_two_smooth_data(seed=29, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.2, 1.2, n)
    mu = 0.25 + 0.45 * x - 0.2 * np.cos(1.3 * z)
    rho = np.full(n, -0.4)
    xi = np.full(n, 0.1)
    y = _sample_gev_response(rng, mu, rho, xi)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _sample_shash_response(rng, mu, sigma, eps, delta):
    z = rng.standard_normal(np.asarray(mu).shape[0])
    return mu + (delta * sigma) * np.sinh((1.0 / delta) * np.arcsinh(z) + eps / delta)


def _shashlss_data(n=120, seed=4):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.5 + 0.8 * x
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    y = _sample_shash_response(rng, mu, sigma, eps, delta)
    return pd.DataFrame({"y": y, "x": x})


def _shashlss_by_data(seed=30, n=160):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    z = rng.uniform(0.5, 1.5, size=n)
    mu = 0.4 + 0.7 * x * z
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    y = _sample_shash_response(rng, mu, sigma, eps, delta)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _shashlss_tensor_data(seed=31, n=160):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = 0.45 + 0.6 * np.sin(np.pi * x0) + 0.2 * x1**2
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    y = _sample_shash_response(rng, mu, sigma, eps, delta)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _shashlss_two_smooth_data(seed=32, n=140):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.2, 1.2, n)
    mu = 0.45 + 0.6 * x - 0.25 * np.cos(1.4 * z)
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    y = _sample_shash_response(rng, mu, sigma, eps, delta)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _gaulss_two_smooth_data(seed=33, n=140):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.2, 1.2, n)
    mu = 0.25 + np.sin(np.pi * x) - 0.35 * np.cos(1.3 * z)
    sigma = np.exp(-0.35 + 0.15 * x + 0.1 * z)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _gammals_two_smooth_data(seed=34, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.2, 1.2, n)
    mu = np.exp(0.35 + 0.3 * x - 0.2 * np.cos(1.4 * z))
    phi = np.exp(-0.5)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _ziplss_two_smooth_data(seed=35, n=140):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    gamma = 0.2 + 0.4 * x - 0.25 * np.cos(1.3 * z)
    eta = -0.3 + 0.5 * z
    y = _sample_ziplss_response(rng, gamma, eta)
    return pd.DataFrame({"y": y, "x": x, "z": z})


GAULSS_FORMULA = ['y ~ s(x, bs="cr", k=6)', "~ 1"]


GENERAL_SE_CASES = [
    ("gaulss_cr", "gaulss", GAULSS_FORMULA, _gaulss_data, "ML", 5e-6, 5e-6, True),
    (
        "gaulss_select_true_cr",
        "gaulss",
        GAULSS_FORMULA,
        _gaulss_data,
        "ML",
        5e-6,
        5e-6,
        True,
    ),
    (
        "gaulss_numeric_by",
        "gaulss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gaulss_by_data,
        "ML",
        5e-6,
        5e-6,
        True,
    ),
    (
        "gaulss_t2_full_false",
        "gaulss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _gaulss_tensor_data,
        "ML",
        2e-4,
        2e-4,
        True,
    ),
    (
        "gaulss_t2_full_true",
        "gaulss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _gaulss_tensor_data,
        "ML",
        2e-4,
        2e-4,
        True,
    ),
    (
        "gaulss_two_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gaulss_two_smooth_data,
        "ML",
        5e-6,
        5e-6,
        True,
    ),
    (
        "gammals_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        1e-5,
        1e-5,
        False,
    ),
    (
        "gammals_select_true_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        1e-5,
        1e-5,
        False,
    ),
    (
        "gammals_numeric_by",
        "gammals",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gammals_by_data,
        "ML",
        2e-5,
        2e-5,
        False,
    ),
    (
        "gammals_t2_full_false",
        "gammals",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _gammals_tensor_data,
        "ML",
        5e-4,
        5e-4,
        False,
    ),
    (
        "gammals_t2_full_true",
        "gammals",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _gammals_tensor_data,
        "ML",
        5e-4,
        5e-4,
        False,
    ),
    (
        "gammals_two_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gammals_two_smooth_data,
        "ML",
        2e-5,
        2e-5,
        False,
    ),
    (
        "gevlss_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "ML",
        2e-5,
        2e-5,
        True,
    ),
    (
        "gevlss_select_true_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "ML",
        2e-5,
        2e-5,
        True,
    ),
    (
        "gevlss_numeric_by",
        "gevlss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_by_data,
        "ML",
        3e-5,
        3e-5,
        True,
    ),
    (
        "gevlss_t2_full_false",
        "gevlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1", "~ 1"],
        _gevlss_tensor_data,
        "ML",
        5e-4,
        5e-4,
        True,
    ),
    (
        "gevlss_t2_full_true",
        "gevlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1", "~ 1"],
        _gevlss_tensor_data,
        "ML",
        5e-4,
        5e-4,
        True,
    ),
    (
        "gevlss_two_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_two_smooth_data,
        "ML",
        3e-5,
        3e-5,
        True,
    ),
    (
        "shashlss_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "ML",
        5e-5,
        5e-5,
        True,
    ),
    (
        "shashlss_select_true_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "ML",
        5e-5,
        5e-5,
        True,
    ),
    (
        "shashlss_numeric_by",
        "shashlss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_by_data,
        "ML",
        8e-5,
        8e-5,
        True,
    ),
    (
        "shashlss_t2_full_false",
        "shashlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1", "~ 1", "~ 1"],
        _shashlss_tensor_data,
        "ML",
        8e-4,
        8e-4,
        True,
    ),
    (
        "shashlss_t2_full_true",
        "shashlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1", "~ 1", "~ 1"],
        _shashlss_tensor_data,
        "ML",
        8e-4,
        8e-4,
        True,
    ),
    (
        "shashlss_two_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_two_smooth_data,
        "ML",
        8e-5,
        8e-5,
        True,
    ),
    (
        "ziplss_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "ML",
        1e-5,
        1e-5,
        False,
    ),
    (
        "ziplss_select_true_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "ML",
        1e-5,
        1e-5,
        False,
    ),
    (
        "ziplss_numeric_by",
        "ziplss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _ziplss_by_data,
        "ML",
        2e-5,
        2e-5,
        False,
    ),
    (
        "ziplss_t2_full_false",
        "ziplss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _ziplss_tensor_data,
        "ML",
        6e-4,
        6e-4,
        False,
    ),
    (
        "ziplss_t2_full_true",
        "ziplss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _ziplss_tensor_data,
        "ML",
        6e-4,
        6e-4,
        False,
    ),
    (
        "ziplss_two_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _ziplss_two_smooth_data,
        "ML",
        2e-5,
        2e-5,
        False,
    ),
]


_GENERAL_FAMILIES = {"gaulss", "gammals", "gevlss", "shashlss", "ziplss"}


def test_general_family_se_case_matrix_covers_requested_surface():
    families = {case[1] for case in GENERAL_SE_CASES}
    assert families >= _GENERAL_FAMILIES

    for family in _GENERAL_FAMILIES:
        family_cases = [case for case in GENERAL_SE_CASES if case[1] == family]
        ids = {case[0] for case in family_cases}
        assert any(case_id.endswith("_cr") for case_id in ids)
        assert any("select_true" in case_id for case_id in ids)
        assert any("numeric_by" in case_id for case_id in ids)
        assert any("t2_full_false" in case_id for case_id in ids)
        assert any("t2_full_true" in case_id for case_id in ids)
        assert any("two_cr" in case_id for case_id in ids)


def _reshape_expected_like(actual, expected):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if expected_arr.shape != actual_arr.shape and expected_arr.size == actual_arr.size:
        expected_arr = expected_arr.reshape(actual_arr.shape, order="F")
    return actual_arr, expected_arr


@pytest.mark.parametrize("method", ["ML", "LAML"])
def test_gaulss_fixed_sp_outer_derivatives_match_mgcv(method):
    data = _gaulss_data()
    expected = _run_mgcv_snapshot(data, GAULSS_FORMULA, "gaulss", method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)

    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", sp)

    actual = float(criterion_value(gam, gam.y_, log_sp, method=method.lower()))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=2e-8,
        rtol=2e-8,
    )

    grad = np.asarray(
        criterion_gradient(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=3e-4,
        rtol=3e-4,
    )

    hess = np.asarray(
        criterion_hessian(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=3e-3,
        rtol=3e-3,
    )


def test_gaulss_sandwich_vcov_matches_mgcv_snapshot():
    data = _gaulss_data(seed=13)
    expected = _run_mgcv_snapshot(data, GAULSS_FORMULA, "gaulss", "REML")
    gam = _fit_nampy_model(data, GAULSS_FORMULA, "gaulss", "REML")

    actual_bayes = np.asarray(gam.vcov(sandwich=True), dtype=np.float64)
    actual_freq = np.asarray(gam.vcov(sandwich=True, freq=True), dtype=np.float64)

    np.testing.assert_allclose(
        actual_bayes,
        np.asarray(expected["fit"]["cov_sandwich_bayes"], dtype=np.float64),
        atol=2e-7,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        actual_freq,
        np.asarray(expected["fit"]["cov_sandwich_freq"], dtype=np.float64),
        atol=2e-7,
        rtol=2e-7,
    )


def test_gaulss_reml_outer_fit_matches_mgcv_without_abnormal_warning():
    data = _gaulss_data(seed=17, n=100)
    expected = _run_mgcv_snapshot(data, GAULSS_FORMULA, "gaulss", "REML")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = _fit_nampy_model(data, GAULSS_FORMULA, "gaulss", "REML")

    abnormal = [
        str(w.message)
        for w in caught
        if "Smoothing optimisation did not converge: ABNORMAL" in str(w.message)
    ]
    assert abnormal == []

    np.testing.assert_allclose(
        np.asarray(np.log(gam.smoothing_params), dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=2e-2,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        float(gam.smoothing_score_),
        float(expected["fit"]["criterion_value"]),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "method", "log_sp_atol", "score_atol"),
    [
        (
            "gevlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            _gevlss_data,
            "ML",
            5e-5,
            5e-6,
        ),
        (
            "shashlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            _shashlss_data,
            "ML",
            8e-2,
            5e-5,
        ),
    ],
)
def test_general_family_higher_order_outer_fit_matches_mgcv_endpoint(
    family, formula, data_factory, method, log_sp_atol, score_atol
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = _fit_nampy_model(data, formula, family, method)

    abnormal = [
        str(w.message)
        for w in caught
        if "Smoothing optimisation did not converge: ABNORMAL" in str(w.message)
    ]
    assert abnormal == []

    np.testing.assert_allclose(
        np.asarray(np.log(gam.smoothing_params), dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=log_sp_atol,
        rtol=log_sp_atol,
    )
    np.testing.assert_allclose(
        float(gam.smoothing_score_),
        float(expected["fit"]["criterion_value"]),
        atol=score_atol,
        rtol=score_atol,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "method", "grad_tol", "hess_tol"),
    [
        ("gammals", ['y ~ s(x, bs="cr", k=6)', "~ 1"], _gammals_data, "ML", 5e-4, 5e-3),
        (
            "gammals",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gammals_data,
            "LAML",
            5e-4,
            5e-3,
        ),
        ("ziplss", ['y ~ s(x, bs="cr", k=6)', "~ 1"], _ziplss_data, "ML", 1e-3, 5e-2),
    ],
)
def test_general_family_fixed_sp_outer_derivatives_match_mgcv(
    family, formula, data_factory, method, grad_tol, hess_tol
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    actual = float(criterion_value(gam, gam.y_, log_sp, method=method.lower()))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=2e-7,
        rtol=2e-7,
    )

    grad = np.asarray(
        criterion_gradient(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=grad_tol,
        rtol=grad_tol,
    )

    hess = np.asarray(
        criterion_hessian(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=hess_tol,
        rtol=hess_tol,
    )

    response = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    np.testing.assert_allclose(
        response.ravel(order="F"),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=3e-6,
        rtol=3e-6,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "method", "grad_tol", "hess_tol"),
    [
        (
            "gevlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            _gevlss_data,
            "ML",
            5e-4,
            5e-3,
        ),
        (
            "shashlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            _shashlss_data,
            "ML",
            5e-4,
            5e-3,
        ),
    ],
)
def test_general_family_fixed_sp_outer_derivatives_match_mgcv_for_higher_order_families(
    family, formula, data_factory, method, grad_tol, hess_tol
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    actual = float(criterion_value(gam, gam.y_, log_sp, method=method.lower()))
    np.testing.assert_allclose(
        actual,
        float(expected["fit"]["criterion_value"]),
        atol=2e-7,
        rtol=2e-7,
    )

    grad = np.asarray(
        criterion_gradient(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        grad,
        np.asarray(expected["fit"]["outer_grad"], dtype=np.float64),
        atol=grad_tol,
        rtol=grad_tol,
    )

    hess = np.asarray(
        criterion_hessian(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=hess_tol,
        rtol=hess_tol,
    )

    response = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    np.testing.assert_allclose(
        response.ravel(order="F"),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=3e-6,
        rtol=3e-6,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory"),
    [
        ("gammals", ['y ~ s(x, bs="cr", k=6)', "~ 1"], _gammals_data),
        ("ziplss", ['y ~ s(x, bs="cr", k=6)', "~ 1"], _ziplss_data),
    ],
)
def test_general_family_sandwich_vcov_matches_mgcv_snapshot(
    family, formula, data_factory
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "REML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    actual_bayes = np.asarray(gam.vcov(sandwich=True), dtype=np.float64)
    actual_freq = np.asarray(gam.vcov(sandwich=True, freq=True), dtype=np.float64)

    np.testing.assert_allclose(
        actual_bayes,
        np.asarray(expected["fit"]["cov_sandwich_bayes"], dtype=np.float64),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        actual_freq,
        np.asarray(expected["fit"]["cov_sandwich_freq"], dtype=np.float64),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "vcov_tol", "resid_tol", "residual_types"),
    [
        (
            "gammals",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gammals_data,
            2e-6,
            2e-6,
            ("response", "pearson", "deviance"),
        ),
        (
            "gaulss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gaulss_data,
            2e-7,
            2e-7,
            ("response", "pearson", "deviance"),
        ),
        (
            "gevlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            _gevlss_data,
            2e-6,
            2e-6,
            ("response", "pearson", "deviance"),
        ),
        (
            "shashlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            _shashlss_data,
            2e-6,
            2e-6,
            (),
        ),
        (
            "ziplss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _ziplss_data,
            2e-6,
            2e-6,
            ("response", "deviance"),
        ),
    ],
)
def test_general_family_prediction_residual_and_vcov_parity_surfaces(
    family, formula, data_factory, vcov_tol, resid_tol, residual_types
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    link = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    response = np.asarray(gam.predict(data, type="response"), dtype=np.float64)

    np.testing.assert_allclose(
        link.ravel(order="F"),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=3e-6,
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        response.ravel(order="F"),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=3e-6,
        rtol=3e-6,
    )

    residual_snapshot = expected["parity"]["diagnostics"]["residuals"]
    for resid_type in residual_types:
        expected_values = residual_snapshot[resid_type]
        actual = np.asarray(gam.residuals(type=resid_type), dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            np.asarray(expected_values, dtype=np.float64),
            atol=resid_tol,
            rtol=resid_tol,
        )

    actual_bayes = np.asarray(gam.vcov(sandwich=True), dtype=np.float64)
    actual_freq = np.asarray(gam.vcov(sandwich=True, freq=True), dtype=np.float64)

    np.testing.assert_allclose(
        actual_bayes,
        np.asarray(expected["fit"]["cov_sandwich_bayes"], dtype=np.float64),
        atol=vcov_tol,
        rtol=vcov_tol,
    )
    np.testing.assert_allclose(
        actual_freq,
        np.asarray(expected["fit"]["cov_sandwich_freq"], dtype=np.float64),
        atol=vcov_tol,
        rtol=vcov_tol,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "atol", "rtol", "compare_cols"),
    [
        (
            "gaulss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gaulss_data,
            1e-7,
            1e-7,
            slice(None),
        ),
        (
            "gevlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            _gevlss_data,
            8e-1,
            2e-2,
            slice(0, 3),
        ),
        (
            "shashlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            _shashlss_data,
            1e-2,
            1e-2,
            slice(None),
        ),
    ],
)
def test_general_family_anova_smooth_parity(
    family, formula, data_factory, atol, rtol, compare_cols
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    actual = gam.anova(freq=False)
    actual_labels = actual.smooth_table["label"].tolist()
    expected_block = expected["parity"]["diagnostics"]["anova_smooth"]

    assert len(actual_labels) == 1
    assert expected_block["labels"] == "s(x)"
    assert actual_labels[0].startswith('s(x, bs="cr"')

    actual_values = np.asarray(
        actual.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(),
        dtype=np.float64,
    )
    expected_values = np.asarray(expected_block["values"], dtype=np.float64)

    np.testing.assert_allclose(
        actual_values[:, compare_cols],
        expected_values[:, compare_cols],
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory"),
    [
        ("gaulss", ['y ~ s(x, bs="cr", k=6)', "~ 1"], _gaulss_data),
        ("gevlss", ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"], _gevlss_data),
        ("shashlss", ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"], _shashlss_data),
    ],
)
def test_general_family_predict_rejects_unimplemented_surfaces(
    family, formula, data_factory
):
    data = data_factory()
    gam = GAM(family=family, formula=formula, optimize_smoothing=False)
    gam.fit(data=data)

    link, link_se = gam.predict(data, type="link", return_se=True)
    response, response_se = gam.predict(data, type="response", return_se=True)
    terms, terms_se = gam.predict(data, type="terms", return_se=True)
    lpmatrix = gam.predict(data, type="lpmatrix")
    shifted = gam.predict(
        data,
        type="link",
        offset=np.full(len(data), 0.25, dtype=np.float64),
    )

    assert np.asarray(link).shape == np.asarray(link_se).shape
    assert np.asarray(response).shape == np.asarray(response_se).shape
    assert np.asarray(terms).shape == np.asarray(terms_se).shape
    assert np.asarray(lpmatrix).shape[0] == len(data)
    np.testing.assert_allclose(
        np.asarray(shifted, dtype=np.float64)[:, 0],
        np.asarray(link, dtype=np.float64)[:, 0] + 0.25,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "family",
        "formula",
        "data_factory",
        "method",
        "pred_atol",
        "se_atol",
        "check_response_se",
    ),
    GENERAL_SE_CASES,
    ids=[case[0] for case in GENERAL_SE_CASES],
)
def test_general_family_link_response_standard_errors_match_mgcv_snapshot(
    case_id,
    family,
    formula,
    data_factory,
    method,
    pred_atol,
    se_atol,
    check_response_se,
):
    select = "select_true" in case_id
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method, select=select)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)

    link, link_se = gam.predict(data, type="link", return_se=True)
    response, response_se = gam.predict(data, type="response", return_se=True)
    link_arr, expected_link = _reshape_expected_like(
        link, expected["predictions"]["link"]
    )
    response_arr, expected_response = _reshape_expected_like(
        response,
        expected["predictions"]["response"],
    )
    link_se_arr, expected_link_se = _reshape_expected_like(
        link_se,
        expected["predictions"]["se_link"],
    )
    response_se_arr, expected_response_se = _reshape_expected_like(
        response_se,
        expected["predictions"]["se_response"],
    )

    np.testing.assert_allclose(
        link_arr,
        expected_link,
        atol=pred_atol,
        rtol=pred_atol,
    )
    np.testing.assert_allclose(
        response_arr,
        expected_response,
        atol=pred_atol,
        rtol=pred_atol,
    )
    np.testing.assert_allclose(
        link_se_arr,
        expected_link_se,
        atol=se_atol,
        rtol=se_atol,
    )
    if check_response_se:
        np.testing.assert_allclose(
            response_se_arr,
            expected_response_se,
            atol=se_atol,
            rtol=se_atol,
        )


def test_shashlss_explicit_unsupported_surfaces_raise():
    data = _shashlss_data()
    gam = GAM(
        family="shashlss",
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)

    with pytest.warns(
        RuntimeWarning,
        match="Pearson residuals not available for this family - returning deviance residuals",
    ), pytest.raises(
        NotImplementedError,
        match="Residual type 'deviance' is not implemented for general family 'shashlss'",
    ):
        gam.residuals(type="pearson")

    with pytest.raises(
        NotImplementedError,
        match="Residual type 'deviance' is not implemented for general family 'shashlss'",
    ):
        gam.k_check(subsample=120, n_rep=8, seed=0)


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "method", "pred_atol", "sp_log_atol"),
    [
        (
            "gaulss",
            GAULSS_FORMULA,
            _gaulss_data,
            "ML",
            5e-6,
            5e-5,
        ),
        (
            "gammals",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gammals_data,
            "ML",
            1e-4,
            2e-4,
        ),
        (
            "gevlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            _gevlss_data,
            "ML",
            2e-5,
            3e-4,
        ),
        (
            "shashlss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            _shashlss_data,
            "ML",
            5e-5,
            8e-4,
        ),
        (
            "ziplss",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _ziplss_data,
            "ML",
            5e-4,
            1e-3,
        ),
    ],
)
def test_general_family_fixed_sp_snapshot_parity_matches_mgcv(
    family, formula, data_factory, method, pred_atol, sp_log_atol
):
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    actual = gam.parity_snapshot(X=data, include_covariances=True)

    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=pred_atol,
        pred_rtol=0.0,
        sp_log_atol=sp_log_atol,
        check_criterion=False,
    )
