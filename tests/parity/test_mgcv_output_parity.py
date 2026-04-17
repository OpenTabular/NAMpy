from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_binomial_data,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data,
    _make_sz_data,
    _run_mgcv_anova,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)


def _make_cyclic_data(seed=77, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2 * np.pi, size=n)
    y = np.sin(x) + 0.3 * np.cos(2 * x) + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_ps_data(seed=81, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(1.3 * x) + 0.2 * x**2 + rng.normal(scale=0.14, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_gp_data(seed=91, n=160):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    y = np.exp(-0.5 * x**2) + 0.4 * np.sin(x) + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_gaussian_univariate_data(seed=301, n=150):
    return _make_gaussian_data(seed=seed, n=n)[["y", "x0"]].rename(columns={"x0": "x"})


def _make_tp_ts_data(seed=111, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(0.8 * x0) + 0.35 * x0 * x1 + 0.2 * x1**2 + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_numeric_by_data(seed=101, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(x) * z + 0.2 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _make_factor_by_data(seed=107, n=240):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    shifts = {"a": 0.6, "b": -0.35, "c": 0.1}
    slopes = {"a": 1.0, "b": -0.7, "c": 0.4}
    y = (
        np.array([shifts[str(v)] for v in f], dtype=np.float64)
        + np.sin(x) * np.array([slopes[str(v)] for v in f], dtype=np.float64)
        + rng.normal(0.0, 0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "f": f})


TERMS_PARITY_CASES = [
    {
        "case_id": "cr",
        "data_factory": lambda: _make_gaussian_univariate_data(seed=301, n=150),
        "formula": 'y ~ s(x, bs="cr", k=8, sp=0.8)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "cs",
        "data_factory": lambda: _make_gaussian_univariate_data(seed=302, n=150),
        "formula": 'y ~ s(x, bs="cs", k=8, sp=1.1)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "cc",
        "data_factory": lambda: _make_cyclic_data(seed=303, n=170),
        "formula": 'y ~ s(x, bs="cc", k=9, sp=0.8)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "ps",
        "data_factory": lambda: _make_ps_data(seed=304, n=170),
        "formula": 'y ~ s(x, bs="ps", k=12, sp=0.5)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "tp",
        "data_factory": lambda: _make_tp_ts_data(seed=305, n=180),
        "formula": 'y ~ s(x0, x1, bs="tp", k=15, sp=1.1)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "ts",
        "data_factory": lambda: _make_tp_ts_data(seed=306, n=180),
        "formula": 'y ~ s(x0, x1, bs="ts", k=15, sp=1.1)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "gp",
        "data_factory": lambda: _make_gp_data(seed=307, n=150),
        "formula": 'y ~ s(x, bs="gp", k=10, sp=1.0)',
        "method": "fixed",
        "pred_atol": 1e-8,
        "pred_rtol": 1e-8,
        "se_atol": 1e-8,
        "se_rtol": 1e-8,
    },
    {
        "case_id": "re",
        "data_factory": _make_random_effect_data,
        "formula": 'y ~ s(f, bs="re", sp=1.0)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "fs",
        "data_factory": _make_fs_data,
        "formula": 'y ~ s(f, x, bs="fs", k=6)',
        "method": "REML",
        "pred_atol": 6e-3,
        "pred_rtol": 6e-3,
        "se_atol": 6e-4,
        "se_rtol": 1e-3,
    },
    {
        "case_id": "sz",
        "data_factory": _make_sz_data,
        "formula": 'y ~ s(f1, f2, x, bs="sz", k=6)',
        "method": "REML",
        "pred_atol": 6e-3,
        "pred_rtol": 6e-3,
        "se_atol": 2e-2,
        "se_rtol": 2e-2,
    },
    {
        "case_id": "mrf",
        "data_factory": _make_mrf_data,
        "formula": (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        ),
        "method": "REML",
        "pred_atol": 6e-3,
        "pred_rtol": 6e-3,
        "se_atol": 6e-3,
        "se_rtol": 6e-3,
    },
    {
        "case_id": "numeric_by_cr",
        "data_factory": _make_numeric_by_data,
        "formula": 'y ~ s(x, by=z, bs="cr", k=8)',
        "method": "REML",
        "pred_atol": 1e-8,
        "pred_rtol": 1e-8,
        "se_atol": 1e-8,
        "se_rtol": 1e-8,
    },
    {
        "case_id": "te",
        "data_factory": lambda: _make_gaussian_data(seed=308, n=180),
        "formula": 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "ti",
        "data_factory": lambda: _make_gaussian_data(seed=309, n=180),
        "formula": 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "t2_full_false",
        "data_factory": lambda: _make_gaussian_data(seed=310, n=180),
        "formula": 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], sp=[0.7, 1.3, 0.9])',
        "method": "fixed",
        "pred_atol": 5e-3,
        "pred_rtol": 5e-3,
        "se_atol": 5e-3,
        "se_rtol": 5e-3,
    },
    {
        "case_id": "t2_full_true",
        "data_factory": lambda: _make_gaussian_data(seed=311, n=180),
        "formula": 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)',
        "method": "REML",
        "pred_atol": 5e-3,
        "pred_rtol": 5e-3,
        "se_atol": 5e-3,
        "se_rtol": 5e-3,
    },
]


SE_SNAPSHOT_CASES = [
    (
        "gaussian_two_cr",
        lambda: _make_gaussian_data(seed=311, n=180),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gaussian",
        "REML",
        1e-10,
        1e-10,
    ),
    (
        "binomial_two_cr",
        lambda: _make_binomial_data(seed=312, n=220),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "binomial",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "poisson_two_cr",
        lambda: _make_poisson_data(seed=313, n=220),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "poisson",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gamma_two_cr",
        lambda: _make_gamma_data(seed=314, n=220),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "negbin_two_cr",
        lambda: _make_negbin_data(seed=315, n=240, theta=2.0),
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        {"name": "negbin", "theta": 2.0},
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gaussian_numeric_by_cr",
        _make_numeric_by_data,
        'y ~ s(x, by=z, bs="cr", k=8)',
        "gaussian",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gaussian_factor_by_cr",
        _make_factor_by_data,
        'y ~ f + s(x, by=f, bs="cr", k=8)',
        "gaussian",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gaussian_t2_full_false",
        lambda: _make_gaussian_data(seed=316, n=180),
        'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])',
        "gaussian",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gaussian_t2_full_true",
        lambda: _make_gaussian_data(seed=317, n=180),
        'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)',
        "gaussian",
        "REML",
        1e-8,
        1e-8,
    ),
    (
        "gaussian_te",
        lambda: _make_gaussian_data(seed=318, n=180),
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        1e-5,
        1e-5,
    ),
    (
        "gaussian_ti",
        lambda: _make_gaussian_data(seed=319, n=180),
        'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        2e-4,
        2e-4,
    ),
]


@pytest.mark.parametrize(
    "case_id, family",
    [
        ("gaussian_cr_uni_reml", "gaussian"),
    ],
)
def test_output_parity_anova_model_comparison(case_id, family):
    data = make_parity_case_data(case_id)
    formulas = [
        'y ~ s(x0, bs="cr", k=8)',
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
    ]
    m0 = _fit_nampy_model(data, formulas[0], family, "REML")
    m1 = _fit_nampy_model(data, formulas[1], family, "REML")
    py = m0.anova(m1, test="Chisq")
    r = _run_mgcv_anova(data, formulas, family, "REML", test="Chisq")
    assert py.table.columns.tolist() == [
        "Resid. Df",
        "Resid. Dev",
        "Df",
        "Deviance",
        "Pr(>Chi)",
    ]

    r_vals = r["table"]["values"]
    r_resid_df = np.asarray([float(row[0]) for row in r_vals], dtype=np.float64)
    r_resid_dev = np.asarray([float(row[1]) for row in r_vals], dtype=np.float64)
    r_df = np.asarray([np.nan, float(r_vals[1][2])], dtype=np.float64)
    r_dev = np.asarray([np.nan, float(r_vals[1][3])], dtype=np.float64)
    r_p = np.asarray([np.nan, float(r_vals[1][4])], dtype=np.float64)

    np.testing.assert_allclose(
        py.table["Resid. Df"].to_numpy(dtype=np.float64),
        r_resid_df,
        atol=5e-6,
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        py.table["Resid. Dev"].to_numpy(dtype=np.float64),
        r_resid_dev,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        py.table["Df"].to_numpy(dtype=np.float64),
        r_df,
        atol=5e-6,
        rtol=5e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        py.table["Deviance"].to_numpy(dtype=np.float64),
        r_dev,
        atol=1e-10,
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        py.table["Pr(>Chi)"].to_numpy(dtype=np.float64),
        r_p,
        atol=1e-12,
        rtol=1e-8,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "case_id, family, pred_type, sample_n, sample_seed, fixed_sp_override",
    [
        ("gaussian_cr_uni_reml", "gaussian", "link", 40, 17, None),
        ("gaussian_cr_uni_fixed", "gaussian", "link", 35, 23, [0.8, 1.5]),
    ],
)
def test_output_parity_newdata_predictions_and_standard_errors(
    case_id, family, pred_type, sample_n, sample_seed, fixed_sp_override
):
    case = get_parity_case(case_id)
    train = make_parity_case_data(case_id)
    newdata = train.sample(n=min(sample_n, len(train)), random_state=sample_seed).copy()

    if fixed_sp_override is not None:
        model = _fit_nampy_model_fixed_sp(
            train,
            case.formula,
            case.family,
            smoothing_params=fixed_sp_override,
        )
    else:
        model = _fit_nampy_model(train, case.formula, case.family, case.method)

    actual_pred, actual_se = model.predict(X=newdata, type=pred_type, return_se=True)
    r_result = _run_mgcv_predict_on_newdata(
        train,
        newdata,
        case.formula,
        family=family,
        method=case.method,
        type=pred_type,
        return_se=True,
    )
    expected_pred = np.asarray(r_result["pred"], dtype=np.float64)
    expected_se = np.asarray(r_result["se"], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(actual_pred, dtype=np.float64), expected_pred, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(actual_se, dtype=np.float64), expected_se, atol=1e-7, rtol=1e-7
    )


def test_output_parity_newdata_terms_linked_id():
    train = make_parity_case_data("gaussian_cr_uni_reml")
    formula = 'y ~ s(x0, bs="cr", k=8, id="shared") + s(x1, bs="cr", k=8, id="shared")'
    newdata = train.sample(n=min(30, len(train)), random_state=29).copy()

    model = _fit_nampy_model(train, formula, "gaussian", "REML")
    actual_pred, actual_se = model.predict(X=newdata, type="link", return_se=True)
    r_result = _run_mgcv_predict_on_newdata(
        train,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="link",
        return_se=True,
    )
    expected_pred = np.asarray(r_result["pred"], dtype=np.float64)
    expected_se = np.asarray(r_result["se"], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(actual_pred, dtype=np.float64), expected_pred, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(actual_se, dtype=np.float64), expected_se, atol=1e-7, rtol=1e-7
    )


def test_output_parity_newdata_lpmatrix_gaussian():
    case = get_parity_case("gaussian_cr_uni_reml")
    train = make_parity_case_data(case.case_id)
    newdata = train.sample(n=min(25, len(train)), random_state=41).copy()

    model = _fit_nampy_model(train, case.formula, case.family, case.method)
    actual = np.asarray(model.predict(X=newdata, type="lpmatrix"), dtype=np.float64)
    r_result = _run_mgcv_predict_on_newdata(
        train,
        newdata,
        case.formula,
        family="gaussian",
        method=case.method,
        type="lpmatrix",
    )
    expected = np.asarray(r_result["pred"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("return_se", [False, True], ids=["no_se", "with_se"])
@pytest.mark.parametrize(
    "case",
    TERMS_PARITY_CASES,
    ids=[case["case_id"] for case in TERMS_PARITY_CASES],
)
def test_output_parity_terms(case, return_se):
    train = case["data_factory"]()
    model = _fit_nampy_model(train, case["formula"], "gaussian", case["method"])

    if return_se:
        actual_terms, actual_se = model.predict(X=train, type="terms", return_se=True)
    else:
        actual_terms = model.predict(X=train, type="terms")
        actual_se = None
    r_result = _run_mgcv_predict_on_newdata(
        train,
        train,
        case["formula"],
        family="gaussian",
        method=case["method"],
        type="terms",
        return_se=return_se,
    )

    expected_terms = np.asarray(r_result["pred"], dtype=np.float64)
    actual_terms = np.asarray(actual_terms, dtype=np.float64)

    assert actual_terms.ndim == expected_terms.ndim == 2
    assert actual_terms.shape == expected_terms.shape
    assert np.atleast_1d(r_result["term_names"]).size == actual_terms.shape[1]

    np.testing.assert_allclose(
        actual_terms,
        expected_terms,
        atol=case["pred_atol"],
        rtol=case["pred_rtol"],
    )
    if return_se:
        expected_se = np.asarray(r_result["se"], dtype=np.float64)
        actual_se = np.asarray(actual_se, dtype=np.float64)
        assert (
            actual_terms.shape
            == expected_terms.shape
            == actual_se.shape
            == expected_se.shape
        )
        np.testing.assert_allclose(
            actual_se,
            expected_se,
            atol=case["se_atol"],
            rtol=case["se_rtol"],
        )


def test_output_parity_fixed_sp_gaussian_offset_predictions():
    rng = np.random.default_rng(91)
    n = 140
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    off = rng.normal(scale=0.25, size=n)
    y = off + np.sin(1.1 * x0) + 0.35 * x1**2 + rng.normal(scale=0.12, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "off": off})
    formula = (
        'y ~ offset(off) + s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'
    )

    model = _fit_nampy_model_fixed_sp(
        data, formula, "gaussian", smoothing_params=[0.8, 1.5]
    )
    actual_resp, actual_se_resp = model.predict(X=data, type="response", return_se=True)
    actual_link, actual_se_link = model.predict(X=data, type="link", return_se=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    np.testing.assert_allclose(
        np.asarray(actual_resp, dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(actual_link, dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se_resp, dtype=np.float64),
        np.asarray(expected["predictions"]["se_response"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se_link, dtype=np.float64),
        np.asarray(expected["predictions"]["se_link"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, pred_atol, se_atol",
    SE_SNAPSHOT_CASES,
    ids=[case[0] for case in SE_SNAPSHOT_CASES],
)
def test_output_parity_snapshot_link_and_response_standard_errors(
    case_id, data_factory, formula, family, method, pred_atol, se_atol
):
    del case_id
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    model = _fit_nampy_model_fixed_sp(data, formula, family, smoothing_params=sp)

    actual_resp, actual_se_resp = model.predict(X=data, type="response", return_se=True)
    actual_link, actual_se_link = model.predict(X=data, type="link", return_se=True)

    np.testing.assert_allclose(
        np.asarray(actual_resp, dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_link, dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se_resp, dtype=np.float64),
        np.asarray(expected["predictions"]["se_response"], dtype=np.float64),
        atol=se_atol,
        rtol=se_atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se_link, dtype=np.float64),
        np.asarray(expected["predictions"]["se_link"], dtype=np.float64),
        atol=se_atol,
        rtol=se_atol,
    )
