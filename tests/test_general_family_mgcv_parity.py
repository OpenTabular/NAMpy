from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_snapshot,
)

from nampy.gam import GAM
from nampy.gam.smoothing_selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)

pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


def _gaulss_data(seed=11, n=140):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    mu = 0.3 + np.sin(np.pi * x)
    sigma = np.exp(-0.35 + 0.25 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _gammals_data(n=100, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.4 + 0.3 * x)
    phi = np.exp(-0.5)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x": x})


def _ziplss_data(n=120, seed=1):
    from scipy.stats import poisson

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    gamma = 0.2 + 0.4 * x
    eta = np.full(n, -0.3)
    lam = np.exp(gamma)
    p = 1.0 - np.exp(-np.exp(eta))
    y = np.zeros(n)
    ind = rng.uniform(size=n) < p
    u = rng.uniform(size=ind.sum())
    u = poisson.cdf(0, lam[ind]) + u * (1.0 - poisson.cdf(0, lam[ind]))
    y[ind] = poisson.ppf(np.minimum(u, 1.0 - 1e-12), lam[ind])
    return pd.DataFrame({"y": y, "x": x})


def _gevlss_data(n=90, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = 0.2 + 0.5 * x
    rho = np.full(n, -0.4)
    xi = np.full(n, 0.1)
    u = rng.uniform(size=n)
    y = mu + ((-np.log(u)) ** (-xi) - 1.0) * np.exp(rho) / xi
    return pd.DataFrame({"y": y, "x": x})


def _shashlss_data(n=120, seed=4):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.5 + 0.8 * x
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    z = rng.standard_normal(n)
    y = mu + (delta * sigma) * np.sinh((1.0 / delta) * np.arcsinh(z) + eps / delta)
    return pd.DataFrame({"y": y, "x": x})


GAULSS_FORMULA = ['y ~ s(x, bs="cr", k=6)', "~ 1"]


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

    with pytest.raises(
        NotImplementedError,
        match="return_se=True is not yet implemented for general families",
    ):
        gam.predict(data, type="response", return_se=True)

    with pytest.raises(
        NotImplementedError,
        match="Prediction offsets are not yet implemented for general families",
    ):
        gam.predict(data, type="response", offset=np.zeros(len(data), dtype=np.float64))

    with pytest.raises(
        NotImplementedError,
        match="General-family prediction currently supports type='link' and type='response'",
    ):
        gam.predict(data, type="terms")


def test_shashlss_explicit_unsupported_surfaces_raise():
    data = _shashlss_data()
    gam = GAM(
        family="shashlss",
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)

    with pytest.raises(
        NotImplementedError,
        match="Residual type 'pearson' is not implemented for general family 'shashlss'",
    ):
        gam.residuals(type="pearson")

    with pytest.raises(
        NotImplementedError,
        match="Residual type 'deviance' is not implemented for general family 'shashlss'",
    ):
        gam.k_check(subsample=120, n_rep=8, seed=0)
