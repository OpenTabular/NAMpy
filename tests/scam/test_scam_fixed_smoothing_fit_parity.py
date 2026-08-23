"""Fixed-smoothing coefficient and behavioral parity with ``scam.fit``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.summary import summary_gam
from nampy.gam.splines.shape import (
    build_bivariate_shape_setup,
    build_scop_univariate_setup,
    predict_scop_univariate,
)
from tests.scam.scam_reference_utils import (
    run_scam_ar1_fixed_fit,
    run_scam_fixed_sp_fit,
    run_scam_linear_functional_fixed_fit,
)

_UPSTREAM_BASIS_CODE = {
    "mpiby": "mpiBy",
    "mpdby": "mpdBy",
    "mdcvby": "mdcvBy",
    "mdcxby": "mdcxBy",
    "micvby": "micvBy",
    "micxby": "micxBy",
    "cvby": "cvBy",
    "cxby": "cxBy",
}

_ALL_UNIVARIATE_CODES = [
    "mpi",
    "mpd",
    "mdcv",
    "mdcx",
    "micv",
    "micx",
    "cv",
    "cx",
    "po",
    "dpo",
    "ipo",
    "miso",
    "mifo",
    "cpop",
    *_UPSTREAM_BASIS_CODE,
]


@pytest.mark.parametrize("basis_code", ["mpi", "mpd"])
@pytest.mark.parametrize("positive_transform", ["exp", "softplus"])
def test_gaussian_monotone_fixed_sp_fit_matches_scam(
    basis_code, positive_transform
):
    rng = np.random.default_rng(922)
    x = np.sort(rng.uniform(-2.0, 2.5, size=90))
    signal = 0.4 + 1.2 / (1.0 + np.exp(-1.8 * x))
    if basis_code == "mpd":
        signal = 2.0 - signal
    y = signal + rng.normal(scale=0.045, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    formula = f"y ~ s(x, bs='{basis_code}', k=8, m=2)"
    start = np.array([0.5, -1.2, -1.0, -0.8, -0.6, -0.7, -0.9, -1.1])
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family="gaussian",
        sp=[0.7],
        start=start,
        positive_transform=positive_transform,
    )
    model = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.7],
        positive_transform=positive_transform,
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data)
    actual = model.fit_result()

    np.testing.assert_array_equal(
        actual.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_allclose(
        actual.coef_optimization,
        expected["coefficients"],
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.coef_full,
        expected["coefficients_t"],
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.mu, expected["mu"], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        actual.deviance, expected["deviance"], rtol=2e-8, atol=2e-10
    )
    np.testing.assert_allclose(
        actual.edf_total, expected["trA"], rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(actual.scale, expected["scale"], rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(
        actual.cov_bayes_optimization,
        expected["Vp"],
        rtol=2e-6,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.cov_bayes,
        expected["Vp_t"],
        rtol=2e-6,
        atol=2e-8,
    )


@pytest.mark.parametrize("basis_code", ["mpi", "mpd", "cv", "cx", "po"])
@pytest.mark.parametrize("order", [1, 2])
def test_univariate_shape_derivative_and_se_match_scam(basis_code, order):
    rng = np.random.default_rng(925)
    x = np.sort(rng.uniform(-1.8, 2.4, size=88))
    setup = build_scop_univariate_setup(
        x, basis_code=basis_code, bs_dim=8, spline_order=2
    )
    start = np.concatenate([[0.24], np.linspace(-1.4, -0.55, setup.n_coef)])
    coef = np.exp(start[1:])
    y = 0.24 + setup.basis_train @ coef + rng.normal(scale=0.015, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    formula = f"y ~ s(x, bs='{basis_code}', k=8, m=2)"
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family="gaussian",
        sp=[0.35],
        start=start,
        include_behavior=True,
    )
    model = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.35],
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data)

    result = model.derivative(smooth_number=1, deriv=order)
    np.testing.assert_allclose(
        result.derivative, expected[f"derivative{order}"], rtol=3e-7, atol=3e-8
    )
    np.testing.assert_allclose(
        result.se, expected[f"derivative{order}_se"], rtol=3e-6, atol=3e-8
    )


def test_shape_prediction_uncertainty_and_residuals_match_scam():
    rng = np.random.default_rng(926)
    x = np.sort(rng.uniform(-1.6, 2.2, size=91))
    setup = build_scop_univariate_setup(
        x, basis_code="mpi", bs_dim=8, spline_order=2
    )
    start = np.concatenate([[0.19], np.linspace(-1.35, -0.5, setup.n_coef)])
    y = 0.19 + setup.basis_train @ np.exp(start[1:])
    y += rng.normal(scale=0.025, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    formula = "y ~ s(x, bs='mpi', k=8, m=2)"
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family="gaussian",
        sp=[0.42],
        start=start,
        include_behavior=True,
    )
    model = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.42],
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data)

    for prediction_type in ("link", "response"):
        fit, se = model.predict(data, type=prediction_type, return_se=True)
        np.testing.assert_allclose(
            fit, expected[f"predict_{prediction_type}"], rtol=3e-8, atol=3e-9
        )
        np.testing.assert_allclose(
            se,
            expected[f"predict_{prediction_type}_se"],
            rtol=3e-6,
            atol=3e-8,
        )
    terms, terms_se = model.predict(data, type="terms", return_se=True)
    np.testing.assert_allclose(terms, expected["predict_terms"], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(
        terms_se, expected["predict_terms_se"], rtol=3e-6, atol=3e-8
    )
    for residual_type in (
        "deviance",
        "pearson",
        "scaled.pearson",
        "working",
        "response",
        "rquantile",
    ):
        actual = model.residuals(type=residual_type, setseed=314)
        np.testing.assert_allclose(
            actual,
            expected[f"residual_{residual_type.replace('.', '_')}"],
            rtol=3e-7,
            atol=3e-8,
        )

    summary = summary_gam(model)
    expected_summary = expected["summary"]
    np.testing.assert_allclose(
        summary.p_table.to_numpy(dtype=np.float64),
        expected_summary["p_table"],
        rtol=3e-6,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        summary.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=np.float64
        ),
        expected_summary["s_table"],
        rtol=2e-4,
        atol=2e-5,
    )
    assert summary.residual_df == pytest.approx(
        expected_summary["residual_df"], rel=3e-7
    )
    assert summary.scale == pytest.approx(expected_summary["scale"], rel=3e-7)
    assert summary.r_sq == pytest.approx(expected_summary["r_sq"], abs=5e-5)
    assert summary.dev_expl == pytest.approx(
        expected_summary["dev_expl"], rel=3e-7
    )
    assert summary.n == expected_summary["n"]
    assert summary.np == expected_summary["np"]


def test_gaussian_shape_ar1_fit_and_standardized_residuals_match_scam():
    rng = np.random.default_rng(927)
    n = 108
    x = np.linspace(-1.8, 2.3, n)
    start = np.concatenate([[0.22], np.linspace(-1.5, -0.65, 7)])
    x_shift = x - float(np.min(x))
    mean = 0.22 + 0.11 * x_shift + 0.025 * x_shift**2
    ar_start = np.zeros(n, dtype=bool)
    ar_start[[0, 54]] = True
    errors = np.empty(n)
    rho = 0.58
    for index in range(n):
        innovation = rng.normal(scale=0.025)
        if index == 0 or ar_start[index]:
            errors[index] = innovation
        else:
            errors[index] = rho * errors[index - 1] + innovation
    data = pd.DataFrame({"y": mean + errors, "x": x})
    formula = "y ~ s(x, bs='mpi', k=8, m=2)"
    expected = run_scam_ar1_fixed_fit(
        data,
        formula,
        sp=[0.38],
        start=start,
        ar1_rho=rho,
        ar_start=ar_start,
    )
    actual_model = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.38],
        start=start,
        ar1_rho=rho,
        ar_start=ar_start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data)
    actual = actual_model.fit_result()

    np.testing.assert_allclose(
        actual.coef_optimization, expected["coefficients"], rtol=4e-7, atol=4e-8
    )
    np.testing.assert_allclose(actual.coef_full, expected["coefficients_t"], rtol=4e-7)
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=4e-8, atol=4e-9)
    np.testing.assert_allclose(actual.deviance, expected["deviance"], rtol=4e-8)
    np.testing.assert_allclose(actual.edf_total, expected["trA"], rtol=4e-7)
    np.testing.assert_allclose(actual.scale, expected["scale"], rtol=4e-7)
    np.testing.assert_allclose(
        actual.cov_bayes, expected["Vp_t"], rtol=4e-6, atol=4e-8
    )
    np.testing.assert_allclose(
        actual_model.ar1_standardized_residuals(),
        expected["std_rsd"],
        rtol=4e-7,
        atol=4e-8,
    )


def test_ar1_observation_transform_is_shared_by_ordinary_gaussian_models():
    data = pd.DataFrame(
        {
            "y": np.linspace(-0.2, 0.5, 20),
            "x": np.linspace(0.0, 1.0, 20),
        }
    )
    model = GAM(
        formula="y ~ x",
        family="gaussian",
        ar1_rho=0.4,
    )
    model.fit(data=data)

    assert model.gam_result_.compiled_model.observation_transform.is_identity is False
    standardized = model.ar1_standardized_residuals()
    assert standardized.shape == (len(data),)
    assert np.all(np.isfinite(standardized))


def test_gaussian_shape_linear_functional_fixed_sp_fit_matches_scam():
    rng = np.random.default_rng(923)
    n, points = 70, 21
    locations = np.tile(np.linspace(-1.5, 2.4, points), (n, 1))
    weights = rng.normal(size=(n, points))
    coefficient = -2.0 / (1.0 + np.exp(-2.0 * locations))
    y = np.sum(weights * coefficient, axis=1) + rng.normal(scale=0.1, size=n)
    data = pd.DataFrame({"y": y, "X": list(locations), "L": list(weights)})
    start = np.array([0.1, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4])
    expected = run_scam_linear_functional_fixed_fit(
        locations,
        weights,
        y,
        basis_code="mpdBy",
        k=8,
        m=2,
        sp=0.4,
        start=start,
    )
    model = GAM(
        formula='y ~ s(X, by=L, bs="mpdby", k=8, m=2)',
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.4],
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data)
    actual = model.fit_result()

    np.testing.assert_allclose(
        actual.coef_optimization, expected["coefficients"], rtol=3e-7, atol=3e-8
    )
    np.testing.assert_allclose(actual.coef_full, expected["coefficients_t"], rtol=3e-7)
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(actual.mu, expected["mu"], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(actual.deviance, expected["deviance"], rtol=3e-8)
    np.testing.assert_allclose(actual.edf_total, expected["trA"], rtol=3e-7)


@pytest.mark.parametrize(
    "basis_code",
    [
        "tedmi",
        "tedmd",
        "temicx",
        "temicv",
        "tedecv",
        "tedecx",
        "tecvcv",
        "tecxcx",
        "tecxcv",
        "tescv",
        "tescx",
        "tesmi1",
        "tesmd1",
        "tesmi2",
        "tesmd2",
        "tismi",
        "tismd",
    ],
)
def test_gaussian_bivariate_shape_fixed_sp_fit_matches_scam(basis_code):
    rng = np.random.default_rng(924)
    x = rng.uniform(-1.7, 2.1, size=96)
    z = rng.uniform(-2.0, 1.5, size=x.size)
    setup = build_bivariate_shape_setup(
        x,
        z,
        basis_code=basis_code,
        bs_dim=(5, 6),
        spline_order=(2, 1),
    )
    term_start = np.linspace(-2.2, -1.15, setup.n_coef)
    if basis_code == "tecvcv":
        # Keep this fully-positive double-curvature fit away from the
        # coefficient-space boundary; boundary coordinates are not uniquely
        # identified and belong in invariant prediction tests instead.
        term_start = np.linspace(-1.25, -0.45, setup.n_coef)
    term_start[~setup.positive_mask] = np.linspace(
        -0.12, 0.12, np.count_nonzero(~setup.positive_mask)
    )
    start = np.concatenate([[0.31], term_start])
    term_coef = np.where(setup.positive_mask, np.exp(term_start), term_start)
    y = 0.31 + setup.basis_train @ term_coef
    y += rng.normal(scale=0.012, size=x.size)
    data = pd.DataFrame({"y": y, "x": x, "z": z})
    formula = f"y ~ s(x, z, bs='{basis_code}', k=c(5, 6), m=c(2, 1))"
    smoothing_params = [1e-4, 1e-4] if basis_code == "tecvcv" else [0.45, 0.75]
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family="gaussian",
        sp=smoothing_params,
        start=start,
    )
    actual = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=smoothing_params,
        positive_transform="exp",
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data).fit_result()

    np.testing.assert_array_equal(
        actual.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_allclose(
        actual.coef_optimization, expected["coefficients"], rtol=4e-7, atol=4e-8
    )
    np.testing.assert_allclose(
        actual.coef_full, expected["coefficients_t"], rtol=4e-7, atol=4e-8
    )
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=4e-8, atol=4e-9)
    np.testing.assert_allclose(actual.deviance, expected["deviance"], rtol=4e-8)
    np.testing.assert_allclose(actual.edf_total, expected["trA"], rtol=4e-7)


@pytest.mark.parametrize("basis_code", _ALL_UNIVARIATE_CODES)
def test_all_univariate_shape_bases_match_scam_fixed_sp_fit(basis_code):
    rng = np.random.default_rng(1371)
    x = np.sort(rng.uniform(-1.6, 2.2, size=84))
    setup = build_scop_univariate_setup(
        x, basis_code=basis_code, bs_dim=8, spline_order=2
    )
    basis = predict_scop_univariate(x, setup)
    is_by_basis = basis_code.endswith("by")
    z = rng.uniform(-1.5, 2.0, size=x.size)
    if is_by_basis:
        basis = basis * z[:, None]

    term_start = np.linspace(-1.05, -0.45, setup.n_coef)
    term_start[~setup.positive_mask] = 0.18
    start = np.concatenate([[0.27], term_start])
    term_coef = np.where(
        setup.positive_mask, np.exp(term_start), term_start
    )
    y = 0.27 + basis @ term_coef + rng.normal(scale=0.012, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    by_clause = ""
    if is_by_basis:
        data["z"] = z
        by_clause = ", by=z"

    upstream_code = _UPSTREAM_BASIS_CODE.get(basis_code, basis_code)
    upstream_formula = (
        f"y ~ s(x{by_clause}, bs='{upstream_code}', k=8, m=2)"
    )
    formula = f"y ~ s(x{by_clause}, bs='{basis_code}', k=8, m=2)"
    expected = run_scam_fixed_sp_fit(
        data,
        upstream_formula,
        family="gaussian",
        sp=[0.65],
        start=start,
    )
    actual = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.65],
        positive_transform="exp",
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data).fit_result()

    np.testing.assert_array_equal(
        actual.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_allclose(
        actual.coef_optimization,
        expected["coefficients"],
        rtol=3e-7,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        actual.coef_full,
        expected["coefficients_t"],
        rtol=3e-7,
        atol=3e-8,
    )
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(
        actual.deviance, expected["deviance"], rtol=3e-8, atol=3e-10
    )
    np.testing.assert_allclose(
        actual.edf_total, expected["trA"], rtol=3e-7, atol=3e-8
    )
    np.testing.assert_allclose(
        actual.cov_bayes_optimization,
        expected["Vp"],
        rtol=3e-6,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        actual.cov_bayes,
        expected["Vp_t"],
        rtol=3e-6,
        atol=3e-8,
    )
    payload = actual.to_dict(include_covariances=True)
    np.testing.assert_allclose(payload["coef_optimization"], actual.coef_optimization)
    np.testing.assert_array_equal(
        payload["positive_coefficient_mask"], actual.positive_coefficient_mask
    )
    np.testing.assert_allclose(
        payload["cov_bayes_optimization"], actual.cov_bayes_optimization
    )


@pytest.mark.parametrize("basis_code", ["lmpi", "lipl"])
def test_local_shape_bases_match_scam_fixed_sp_fit(basis_code):
    rng = np.random.default_rng(2044)
    x = np.sort(rng.uniform(-1.9, 3.0, size=91))
    change_point = 0.4
    setup = build_scop_univariate_setup(
        x,
        basis_code=basis_code,
        bs_dim=12,
        spline_order=2,
        change_point=change_point,
    )
    basis = predict_scop_univariate(x, setup)
    term_start = np.linspace(-0.95, -0.35, setup.n_coef)
    term_start[~setup.positive_mask] = np.linspace(
        -0.2, 0.2, np.count_nonzero(~setup.positive_mask)
    )
    start = np.concatenate([[0.22], term_start])
    term_coef = np.where(
        setup.positive_mask, np.exp(term_start), term_start
    )
    y = 0.22 + basis @ term_coef + rng.normal(scale=0.015, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    formula = (
        f"y ~ s(x, bs='{basis_code}', k=12, m=2, "
        f"xt=list(xc={change_point}))"
    )
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family="gaussian",
        sp=[0.72],
        start=start,
    )
    actual = GAM(
        formula=formula,
        family="gaussian",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.72],
        positive_transform="exp",
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data).fit_result()

    np.testing.assert_array_equal(
        actual.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_allclose(
        actual.coef_optimization,
        expected["coefficients"],
        rtol=3e-7,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        actual.coef_full,
        expected["coefficients_t"],
        rtol=3e-7,
        atol=3e-8,
    )
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(
        actual.edf_total, expected["trA"], rtol=3e-7, atol=3e-8
    )
    np.testing.assert_allclose(
        actual.cov_bayes,
        expected["Vp_t"],
        rtol=3e-6,
        atol=3e-8,
    )


@pytest.mark.parametrize(
    "python_family,r_family",
    [
        pytest.param("poisson", "poisson", id="poisson"),
        pytest.param("binomial", "binomial", id="binomial"),
        pytest.param(("gamma", "log"), "Gamma(link='log')", id="gamma-log"),
    ],
)
def test_non_gaussian_shape_newton_matches_scam_fixed_sp(
    python_family, r_family
):
    rng = np.random.default_rng(2281)
    x = np.sort(rng.uniform(-1.8, 2.4, size=320))
    family_name = python_family if isinstance(python_family, str) else python_family[0]
    smooth_signal = -1.6 + 3.8 / (1.0 + np.exp(-1.5 * x))
    if python_family == "poisson":
        y = rng.poisson(np.exp(smooth_signal)).astype(np.float64)
    elif family_name == "binomial":
        probability = 1.0 / (1.0 + np.exp(-smooth_signal))
        y = rng.binomial(1, probability).astype(np.float64)
    else:
        mean = np.exp(smooth_signal)
        shape = 8.0
        y = rng.gamma(shape, mean / shape).astype(np.float64)
    data = pd.DataFrame({"y": y, "x": x})
    formula = "y ~ s(x, bs='mpi', k=8, m=2)"
    start = np.array([0.15, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
    expected = run_scam_fixed_sp_fit(
        data,
        formula,
        family=r_family,
        sp=[0.58],
        start=start,
    )
    actual = GAM(
        formula=formula,
        family=python_family,
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.58],
        positive_transform="exp",
        start=start,
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=data).fit_result()

    np.testing.assert_allclose(
        actual.coef_optimization,
        expected["coefficients"],
        rtol=8e-7,
        atol=8e-8,
    )
    np.testing.assert_allclose(
        actual.coef_full,
        expected["coefficients_t"],
        rtol=8e-7,
        atol=8e-8,
    )
    np.testing.assert_allclose(actual.eta, expected["eta"], rtol=8e-8, atol=8e-9)
    np.testing.assert_allclose(actual.mu, expected["mu"], rtol=8e-8, atol=8e-9)
    np.testing.assert_allclose(
        actual.deviance, expected["deviance"], rtol=8e-8, atol=8e-9
    )
    np.testing.assert_allclose(
        actual.edf_total, expected["trA"], rtol=8e-7, atol=8e-8
    )
