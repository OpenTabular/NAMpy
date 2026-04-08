from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_fs_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_random_effect_data,
    _make_sz_data,
    _run_mgcv_anova,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


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


def _make_tp_ts_data(seed=111, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(0.8 * x0) + 0.35 * x0 * x1 + 0.2 * x1**2 + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


TERMS_PARITY_CASES = [
    {
        "case_id": "cr",
        "data_factory": lambda: _make_gaussian_data(seed=301, n=150)[
            ["y", "x0"]
        ].rename(columns={"x0": "x"}),
        "formula": 'y ~ s(x, bs="cr", k=8, sp=0.8)',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "cs",
        "data_factory": lambda: _make_gaussian_data(seed=302, n=150)[
            ["y", "x0"]
        ].rename(columns={"x0": "x"}),
        "formula": 'y ~ s(x, bs="cs", k=8, sp=1.1)',
        "method": "fixed",
        "pred_atol": 1e-4,
        "pred_rtol": 1e-4,
        "se_atol": 1e-5,
        "se_rtol": 1e-5,
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
        "case_id": "te",
        "data_factory": lambda: _make_gaussian_data(seed=308, n=150),
        "formula": 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "ti",
        "data_factory": lambda: _make_gaussian_data(seed=309, n=150),
        "formula": 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        "method": "fixed",
        "pred_atol": 1e-10,
        "pred_rtol": 1e-10,
        "se_atol": 1e-10,
        "se_rtol": 1e-10,
    },
    {
        "case_id": "t2",
        "data_factory": lambda: _make_gaussian_data(seed=310, n=150),
        "formula": 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])',
        "method": "fixed",
        "pred_atol": 1e-7,
        "pred_rtol": 1e-7,
        "se_atol": 1e-2,
        "se_rtol": 1e-2,
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
        "se_atol": 2e-4,
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
]


@pytest.mark.parametrize(
    "case_id, family",
    [
        ("gaussian_cr_uni_reml", "gaussian"),
        ("poisson_cr_uni_reml", "poisson"),
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
    py_dev = py.table["deviance"].to_numpy(dtype=np.float64)
    r_dev = np.asarray(
        [float(row[1]) for row in r["table"]["values"]], dtype=np.float64
    )
    np.testing.assert_allclose(py_dev, r_dev, atol=1.0, rtol=0.1)


@pytest.mark.parametrize(
    "case_id, family, pred_type",
    [
        ("gaussian_cr_uni_reml", "gaussian", "link"),
        ("poisson_cr_uni_reml", "poisson", "response"),
        ("binomial_cr_uni_reml", "binomial", "link"),
    ],
)
def test_output_parity_newdata_predictions(case_id, family, pred_type):
    case = get_parity_case(case_id)
    train = make_parity_case_data(case_id)
    newdata = train.sample(n=min(40, len(train)), random_state=17).copy()
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


@pytest.mark.parametrize(
    "case_id, family, pred_type",
    [
        ("gaussian_cr_uni_fixed", "gaussian", "link"),
        ("poisson_cr_uni_reml", "poisson", "response"),
    ],
)
def test_output_parity_newdata_standard_errors(case_id, family, pred_type):
    case = get_parity_case(case_id)
    train = make_parity_case_data(case_id)
    newdata = train.sample(n=min(35, len(train)), random_state=23).copy()

    if case.method == "fixed":
        model = _fit_nampy_model_fixed_sp(
            train,
            case.formula,
            case.family,
            smoothing_params=[0.8, 1.5],
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


@pytest.mark.parametrize(
    "case",
    TERMS_PARITY_CASES,
    ids=[case["case_id"] for case in TERMS_PARITY_CASES],
)
def test_output_parity_terms_all_smooth_types(case):
    train = case["data_factory"]()
    model = _fit_nampy_model(train, case["formula"], "gaussian", case["method"])

    actual_terms = model.predict(X=train, type="terms")
    r_result = _run_mgcv_predict_on_newdata(
        train,
        train,
        case["formula"],
        family="gaussian",
        method=case["method"],
        type="terms",
        return_se=False,
    )

    expected_terms = np.asarray(r_result["pred"], dtype=np.float64)
    actual_terms = np.asarray(actual_terms, dtype=np.float64)

    assert actual_terms.ndim == expected_terms.ndim == 2
    assert actual_terms.shape == expected_terms.shape
    assert actual_terms.shape[1] == 1
    assert np.atleast_1d(r_result["term_names"]).size == 1

    np.testing.assert_allclose(
        actual_terms,
        expected_terms,
        atol=case["pred_atol"],
        rtol=case["pred_rtol"],
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in TERMS_PARITY_CASES
        if "se_atol" in case
    ],
    ids=[
        case["case_id"]
        for case in TERMS_PARITY_CASES
        if "se_atol" in case
    ],
)
def test_output_parity_terms_standard_errors(case):
    train = case["data_factory"]()
    model = _fit_nampy_model(train, case["formula"], "gaussian", case["method"])

    actual_terms, actual_se = model.predict(X=train, type="terms", return_se=True)
    r_result = _run_mgcv_predict_on_newdata(
        train,
        train,
        case["formula"],
        family="gaussian",
        method=case["method"],
        type="terms",
        return_se=True,
    )

    expected_terms = np.asarray(r_result["pred"], dtype=np.float64)
    expected_se = np.asarray(r_result["se"], dtype=np.float64)
    actual_terms = np.asarray(actual_terms, dtype=np.float64)
    actual_se = np.asarray(actual_se, dtype=np.float64)

    assert (
        actual_terms.shape
        == expected_terms.shape
        == actual_se.shape
        == expected_se.shape
    )
    assert np.atleast_1d(r_result["term_names"]).size == 1

    np.testing.assert_allclose(
        actual_terms,
        expected_terms,
        atol=case["pred_atol"],
        rtol=case["pred_rtol"],
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
