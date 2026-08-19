from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.model_state import _term_blocks_seq
from nampy.gam.results.snapshots import _normalize_reference_term_label
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _run_mgcv_predict_on_newdata,
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
]


_GENERAL_FAMILIES = {"gaulss", "gammals"}

GENERAL_OUTER_DERIV_TOLS = {
    "gaulss": {"grad": 3e-4, "hess": 3e-3},
    "gammals": {"grad": 5e-4, "hess": 5e-3},
}

GENERAL_OUTER_ENDPOINT_TOLS = {
    "gaulss": {"log_sp": 2e-2, "score": 2e-5},
    "gammals": {"log_sp": 2e-2, "score": 5e-5},
}


def test_general_family_se_case_matrix_covers_requested_surface():
    """
    Verify that general family standard errors case matrix covers requested surface.
    """
    families = {case[1] for case in GENERAL_SE_CASES}
    assert families >= _GENERAL_FAMILIES

    for family in _GENERAL_FAMILIES:
        family_cases = [case for case in GENERAL_SE_CASES if case[1] == family]
        ids = {case[0] for case in family_cases}
        assert any(case_id.endswith("_cr") for case_id in ids)
        assert any("select_true" in case_id for case_id in ids)
        assert any("numeric_by" in case_id for case_id in ids)


def _reshape_expected_like(actual, expected):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if expected_arr.shape != actual_arr.shape and expected_arr.size == actual_arr.size:
        expected_arr = expected_arr.reshape(actual_arr.shape, order="F")
    return actual_arr, expected_arr


def _outer_case_tolerances(case_id: str, family: str):
    deriv = dict(GENERAL_OUTER_DERIV_TOLS[family])
    endpoint = dict(GENERAL_OUTER_ENDPOINT_TOLS[family])

    return deriv, endpoint


def _general_newdata(data: pd.DataFrame, *, n: int = 31) -> pd.DataFrame:
    cols: dict[str, np.ndarray | pd.Series] = {}
    for col in data.columns:
        if col == "y":
            continue
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=np.float64, copy=False)
            lo = float(np.nanquantile(values, 0.15))
            hi = float(np.nanquantile(values, 0.85))
            if not np.isfinite(lo):
                lo = float(np.nanmin(values))
            if not np.isfinite(hi):
                hi = float(np.nanmax(values))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo = hi = 0.0
            if np.isclose(lo, hi):
                cols[col] = np.full(n, lo, dtype=np.float64)
            else:
                cols[col] = np.linspace(lo, hi, n, dtype=np.float64)
            continue
        cols[col] = pd.Series([series.iloc[0]] * n, dtype=series.dtype)
    return pd.DataFrame(cols)


def _general_diag_tol(base_atol: float) -> float:
    return max(1e-5, 10.0 * float(base_atol))


def _general_kcheck_edf_tol(base_atol: float) -> float:
    return max(5e-5, 10.0 * float(base_atol))


def _assert_general_prediction_close(actual, expected, *, atol: float) -> None:
    actual_arr, expected_arr = _reshape_expected_like(actual, expected)
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        atol=atol,
        rtol=atol,
    )


def _assert_general_lpmatrix_close(actual, expected, *, atol: float) -> None:
    actual_arr, expected_arr = _reshape_expected_like(actual, expected)
    if actual_arr.shape == expected_arr.shape:
        signed = actual_arr.copy()
        for j in range(signed.shape[1]):
            direct = float(np.linalg.norm(signed[:, j] - expected_arr[:, j]))
            flipped = float(np.linalg.norm(-signed[:, j] - expected_arr[:, j]))
            if flipped < direct:
                signed[:, j] *= -1.0
        np.testing.assert_allclose(
            signed,
            expected_arr,
            atol=atol,
            rtol=atol,
        )
        return
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        atol=atol,
        rtol=atol,
    )


def _assert_general_endpoint_log_sp_close(
    actual_log_sp, expected_log_sp, *, atol: float
) -> None:
    actual_arr = np.atleast_1d(np.asarray(actual_log_sp, dtype=np.float64))
    expected_arr = np.atleast_1d(np.asarray(expected_log_sp, dtype=np.float64))
    # Very large smoothing parameters are endpoint-flat in these fits; the score
    # assertion below is the behavioral parity check for that saturated tail.
    high_penalty = (actual_arr > 10.0) & (expected_arr > 10.0)
    np.testing.assert_allclose(
        actual_arr[~high_penalty],
        expected_arr[~high_penalty],
        atol=atol,
        rtol=atol,
    )


def _assert_general_term_labels_match(gam: GAM, expected_names) -> None:
    if expected_names is None:
        expected_names = []
    elif isinstance(expected_names, str):
        expected_names = [expected_names]
    actual_labels = [
        _normalize_reference_term_label(getattr(tb, "label", None))
        for tb in _term_blocks_seq(gam)
    ]
    expected_labels = [_normalize_reference_term_label(name) for name in expected_names]
    assert actual_labels == expected_labels


@pytest.mark.parametrize("method", ["ML", "LAML"])
def test_gaulss_fixed_sp_outer_derivatives_match_mgcv(method):
    """Verify that gaulss fixed sp outer derivatives match mgcv."""
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
    """Verify that gaulss sandwich vcov matches mgcv snapshot."""
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
    """Verify that gaulss REML outer fit matches mgcv without abnormal warning."""
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
    (
        "case_id",
        "family",
        "formula",
        "data_factory",
        "method",
        "_pred_atol",
        "_se_atol",
        "_check_response_se",
    ),
    GENERAL_SE_CASES,
    ids=[case[0] for case in GENERAL_SE_CASES],
)
def test_general_family_fixed_sp_outer_derivatives_match_mgcv_across_surface(
    case_id,
    family,
    formula,
    data_factory,
    method,
    _pred_atol,
    _se_atol,
    _check_response_se,
):
    """
    Verify that general family fixed sp outer derivatives match mgcv across surface.
    """
    del _pred_atol, _se_atol, _check_response_se
    select = "select_true" in case_id
    deriv_tol, _endpoint_tol = _outer_case_tolerances(case_id, family)
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method, select=select)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)

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
        atol=deriv_tol["grad"],
        rtol=deriv_tol["grad"],
    )

    hess = np.asarray(
        criterion_hessian(gam, gam.y_, log_sp, method=method.lower()),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        hess,
        np.asarray(expected["fit"]["outer_hess"], dtype=np.float64),
        atol=deriv_tol["hess"],
        rtol=deriv_tol["hess"],
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "family",
        "formula",
        "data_factory",
        "method",
        "_pred_atol",
        "_se_atol",
        "_check_response_se",
    ),
    GENERAL_SE_CASES,
    ids=[case[0] for case in GENERAL_SE_CASES],
)
def test_general_family_outer_fit_matches_mgcv_endpoint_across_surface(
    case_id,
    family,
    formula,
    data_factory,
    method,
    _pred_atol,
    _se_atol,
    _check_response_se,
):
    """Verify that general family outer fit matches mgcv endpoint across surface."""
    del _pred_atol, _se_atol, _check_response_se
    select = "select_true" in case_id
    _deriv_tol, endpoint_tol = _outer_case_tolerances(case_id, family)
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method, select=select)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = _fit_nampy_model(data, formula, family, method, select=select)

    abnormal = [
        str(w.message)
        for w in caught
        if "Smoothing optimisation did not converge: ABNORMAL" in str(w.message)
    ]
    assert abnormal == []

    _assert_general_endpoint_log_sp_close(
        np.asarray(np.log(gam.smoothing_params), dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=endpoint_tol["log_sp"],
    )
    np.testing.assert_allclose(
        float(gam.smoothing_score_),
        float(expected["fit"]["criterion_value"]),
        atol=endpoint_tol["score"],
        rtol=endpoint_tol["score"],
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
    ],
)
def test_general_family_fixed_sp_outer_derivatives_match_mgcv(
    family, formula, data_factory, method, grad_tol, hess_tol
):
    """Verify that general family fixed sp outer derivatives match mgcv."""
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
    ],
)
def test_general_family_sandwich_vcov_matches_mgcv_snapshot(
    family, formula, data_factory
):
    """Verify that general family sandwich vcov matches mgcv snapshot."""
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
    ],
)
def test_general_family_prediction_residual_and_vcov_parity_surfaces(
    family, formula, data_factory, vcov_tol, resid_tol, residual_types
):
    """Verify that general family prediction residual and vcov parity surfaces."""
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
    ],
)
def test_general_family_anova_smooth_parity(
    family, formula, data_factory, atol, rtol, compare_cols
):
    """Verify that general family anova smooth parity."""
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
    ],
)
def test_general_family_predict_rejects_unimplemented_surfaces(
    family, formula, data_factory
):
    """Verify that general family predict rejects unimplemented surfaces."""
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
    """Verify that general family link response standard errors match mgcv snapshot."""
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
@pytest.mark.parametrize(
    "pred_type",
    ["link", "response", "terms", "lpmatrix"],
    ids=["link", "response", "terms", "lpmatrix"],
)
def test_general_family_newdata_prediction_surfaces_match_mgcv(
    case_id,
    family,
    formula,
    data_factory,
    method,
    pred_atol,
    se_atol,
    check_response_se,
    pred_type,
):
    """Verify that general family new-data prediction surfaces match mgcv."""
    select = "select_true" in case_id
    data = data_factory()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(data, formula, family, method, select=select)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method=method,
        type=pred_type,
        return_se=(pred_type != "lpmatrix"),
        select=select,
    )

    if pred_type == "lpmatrix":
        actual = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            np.asarray(expected["pred"], dtype=np.float64),
            atol=max(1e-8, pred_atol),
            rtol=max(1e-8, pred_atol),
        )
        return

    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)
    _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)

    if pred_type == "terms":
        _assert_general_term_labels_match(gam, expected.get("term_names", []))
        _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)
        return

    if pred_type == "response" and not check_response_se:
        assert (
            np.asarray(actual_se, dtype=np.float64).shape
            == np.asarray(actual_pred, dtype=np.float64).shape
        )
        return

    _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)


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
@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
    case_id,
    family,
    formula,
    data_factory,
    method,
    pred_atol,
    se_atol,
    check_response_se,
    pred_type,
):
    """Verify that general family new-data unconditional standard errors match mgcv."""
    select = "select_true" in case_id
    data = data_factory()
    newdata = _general_newdata(data)
    snapshot = _run_mgcv_snapshot(data, formula, family, method, select=select)
    sp = np.asarray(snapshot["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method=method,
        type=pred_type,
        return_se=True,
        unconditional=True,
        select=select,
    )

    actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
    actual_pred, actual_se = gam.predict(
        newdata,
        type=pred_type,
        return_se=True,
        cov=actual_cov,
    )
    _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)

    if pred_type == "terms":
        _assert_general_term_labels_match(gam, expected.get("term_names", []))
        _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)
        return

    if pred_type == "response" and not check_response_se:
        assert (
            np.asarray(actual_se, dtype=np.float64).shape
            == np.asarray(actual_pred, dtype=np.float64).shape
        )
        return

    _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)


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
    ],
)
def test_general_family_fixed_sp_snapshot_parity_matches_mgcv(
    family, formula, data_factory, method, pred_atol, sp_log_atol
):
    """Verify that general family fixed sp snapshot parity matches mgcv."""
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


def test_gammals_reml_outer_fit_matches_mgcv():
    """gammals x REML was the last never-fitted general-family method cell.

    GENERAL_SE_CASES is all-ML; the REML route (folded LAML) goes through the
    same gam.fit5 machinery but was previously unexercised end-to-end.
    """
    data = _gammals_data()
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]

    actual = _fit_nampy_snapshot(data, formula, "gammals", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gammals", "REML")

    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["edf_total"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["edf_total"], dtype=np.float64)),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gaulss_efs_endpoint_matches_mgcv_with_vc_degenerating_to_vb():
    """General-family EFS: mgcv endpoint parity plus the deriv=0 post-proc.

    Upstream gam.fit5.post.proc with efs/optim runs at deriv=0: no
    smoothing-uncertainty correction, so Vc == Vb exactly. EFS is a
    fixed-point iteration, so the endpoint is compared at the level the two
    iteration paths support rather than Newton-strict.
    """
    data = _gaulss_data()
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]

    from nampy.gam import GAM

    gam = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data, formula, "gaulss", "REML", optimizer="efs"
    )

    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=2e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-4,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(gam.vcov(unconditional=True), dtype=np.float64),
        np.asarray(gam.vcov(), dtype=np.float64),
        # Vc is literally Vb under deriv=0 post-processing; the export path
        # symmetrizes one copy, so allow last-ulp noise only.
        atol=1e-15,
        rtol=0.0,
    )
