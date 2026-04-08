from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_anova,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


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
    r_dev = np.asarray([float(row[1]) for row in r["table"]["values"]], dtype=np.float64)
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
    np.testing.assert_allclose(np.asarray(actual_pred, dtype=np.float64), expected_pred, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(actual_se, dtype=np.float64), expected_se, atol=1e-7, rtol=1e-7)


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
    np.testing.assert_allclose(np.asarray(actual_pred, dtype=np.float64), expected_pred, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(actual_se, dtype=np.float64), expected_se, atol=1e-7, rtol=1e-7)


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
    np.testing.assert_allclose(np.asarray(actual_pred, dtype=np.float64), expected_pred, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(actual_se, dtype=np.float64), expected_se, atol=1e-7, rtol=1e-7)


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


def test_output_parity_fixed_sp_gaussian_offset_predictions():
    rng = np.random.default_rng(91)
    n = 140
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    off = rng.normal(scale=0.25, size=n)
    y = off + np.sin(1.1 * x0) + 0.35 * x1 ** 2 + rng.normal(scale=0.12, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "off": off})
    formula = 'y ~ offset(off) + s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    model = _fit_nampy_model_fixed_sp(data, formula, "gaussian", smoothing_params=[0.8, 1.5])
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
