from __future__ import annotations

import numpy as np
import pytest

from tests.families.test_general_family_mgcv_parity import (
    GENERAL_SE_CASES,
    _assert_general_prediction_close,
    _assert_general_term_labels_match,
    _gammals_data,
    _gaulss_data,
    _general_newdata,
)
from tests.mgcv_parity_utils import _fit_nampy_model, _run_mgcv_predict_on_newdata
from tests.optimization.test_mgcv_fixed_inner_fit_parity import (
    _make_linked_id_univariate_data,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]

_PREDICTION_FAMILY = "gaulss"
_PREDICTION_FORMULA = ['y ~ s(x, bs="cr", k=6)', "~ 1"]
_PREDICTION_METHOD = "ML"
_PREDICTION_ATOL = 5e-6
_SELECT_TRUE = True

_GENERAL_CASES_BY_ID = {case[0]: case for case in GENERAL_SE_CASES}
_BROADER_PREDICTION_STAGE_CASE_IDS = [
    "gaulss_cr",
    "gammals_numeric_by",
]
_METHOD_STAGE_CASES = [
    pytest.param(
        "gaulss_laml_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gaulss_data,
        "LAML",
        5e-6,
        5e-6,
        True,
        id="gaulss_laml_cr",
    ),
    pytest.param(
        "gammals_laml_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "LAML",
        1e-5,
        1e-5,
        False,
        id="gammals_laml_cr",
    ),
]
def _load_general_stage_case(case_id):
    (
        _case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
    ) = _GENERAL_CASES_BY_ID[case_id]
    return family, formula, data_factory, method, pred_atol, se_atol, check_response_se


def _assert_stage_prediction_surface(
    gam,
    expected,
    *,
    pred_type: str,
    actual_pred,
    actual_se,
    pred_atol: float,
    se_atol: float,
    check_response_se: bool,
):
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


@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_select_true_newdata_prediction_surfaces_match_mgcv(pred_type):
    """
    Verify that general family select true new-data prediction surfaces match mgcv.
    """
    data = _gaulss_data()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(
        data,
        _PREDICTION_FORMULA,
        _PREDICTION_FAMILY,
        _PREDICTION_METHOD,
        select=_SELECT_TRUE,
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _PREDICTION_FORMULA,
        family=_PREDICTION_FAMILY,
        method=_PREDICTION_METHOD,
        type=pred_type,
        return_se=True,
        select=_SELECT_TRUE,
    )

    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)

    _assert_general_prediction_close(
        actual_pred, expected["pred"], atol=_PREDICTION_ATOL
    )
    if pred_type == "terms":
        _assert_general_term_labels_match(gam, expected.get("term_names", []))
    _assert_general_prediction_close(actual_se, expected["se"], atol=_PREDICTION_ATOL)


@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_unconditional_standard_errors_match_mgcv_on_select_true_case(
    pred_type,
):
    """
    Verify that general family unconditional standard errors match mgcv on select true
    case.
    """
    data = _gaulss_data(seed=13)
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(
        data,
        _PREDICTION_FORMULA,
        _PREDICTION_FAMILY,
        _PREDICTION_METHOD,
        select=_SELECT_TRUE,
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _PREDICTION_FORMULA,
        family=_PREDICTION_FAMILY,
        method=_PREDICTION_METHOD,
        type=pred_type,
        return_se=True,
        unconditional=True,
        select=_SELECT_TRUE,
    )

    actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
    actual_pred, actual_se = gam.predict(
        newdata,
        type=pred_type,
        return_se=True,
        cov=actual_cov,
    )

    _assert_general_prediction_close(
        actual_pred, expected["pred"], atol=_PREDICTION_ATOL
    )
    if pred_type == "terms":
        _assert_general_term_labels_match(gam, expected.get("term_names", []))
    _assert_general_prediction_close(actual_se, expected["se"], atol=_PREDICTION_ATOL)


@pytest.mark.parametrize("case_id", _BROADER_PREDICTION_STAGE_CASE_IDS)
@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_stage_matrix_newdata_prediction_surfaces_match_mgcv(
    case_id,
    pred_type,
):
    """
    Verify that general family stage matrix new-data prediction surfaces match mgcv.
    """
    family, formula, data_factory, method, pred_atol, se_atol, check_response_se = (
        _load_general_stage_case(case_id)
    )
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
        return_se=True,
        select=select,
    )

    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)
    _assert_stage_prediction_surface(
        gam,
        expected,
        pred_type=pred_type,
        actual_pred=actual_pred,
        actual_se=actual_se,
        pred_atol=pred_atol,
        se_atol=se_atol,
        check_response_se=check_response_se,
    )


@pytest.mark.parametrize("case_id", _BROADER_PREDICTION_STAGE_CASE_IDS)
@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_stage_matrix_unconditional_standard_errors_match_mgcv(
    case_id,
    pred_type,
):
    """
    Verify that general family stage matrix unconditional standard errors match mgcv.
    """
    family, formula, data_factory, method, pred_atol, se_atol, check_response_se = (
        _load_general_stage_case(case_id)
    )
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
    _assert_stage_prediction_surface(
        gam,
        expected,
        pred_type=pred_type,
        actual_pred=actual_pred,
        actual_se=actual_se,
        pred_atol=pred_atol,
        se_atol=se_atol,
        check_response_se=check_response_se,
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
    _METHOD_STAGE_CASES,
)
@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_method_stage_prediction_surfaces_match_mgcv(
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
    """Verify that general family method stage prediction surfaces match mgcv."""
    del case_id
    data = data_factory()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(data, formula, family, method)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method=method,
        type=pred_type,
        return_se=True,
    )

    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)
    _assert_stage_prediction_surface(
        gam,
        expected,
        pred_type=pred_type,
        actual_pred=actual_pred,
        actual_se=actual_se,
        pred_atol=pred_atol,
        se_atol=se_atol,
        check_response_se=check_response_se,
    )

@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_linked_id_public_prediction_surfaces_match_mgcv_on_supported_gaussian_case(
    pred_type,
):
    """
    Verify that linked id public prediction surfaces match mgcv on supported gaussian
    case.
    """
    data = _make_linked_id_univariate_data()
    newdata = data.sample(n=min(31, len(data)), random_state=13).copy()
    formula = 'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")'
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type=pred_type,
        return_se=True,
    )

    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)
    _assert_general_prediction_close(actual_pred, expected["pred"], atol=1e-7)
    if pred_type == "terms":
        _assert_general_term_labels_match(gam, expected.get("term_names", []))
    _assert_general_prediction_close(actual_se, expected["se"], atol=1e-7)



def test_general_family_iterms_downgrades_to_terms_with_mgcv_warning():
    """Multi-predictor type="iterms" downgrades to "terms" like predict.gam.

    Upstream predict.gam warns "iterms reset to terms" for multi-linear-
    predictor families; NAMpy mirrors the warning and the downgraded output
    must equal an explicit type="terms" call and mgcv's own type="iterms"
    result on the same newdata.
    """
    import warnings

    data = _gaulss_data()
    gam = _fit_nampy_model(data, _PREDICTION_FORMULA, _PREDICTION_FAMILY, "ML")
    newdata = _general_newdata(data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iterms_pred, iterms_se = gam.predict(
            newdata, type="iterms", return_se=True
        )
    messages = [str(w.message) for w in caught]
    assert any(
        "type iterms not available for multiple predictor cases" in message
        for message in messages
    )

    terms_pred, terms_se = gam.predict(newdata, type="terms", return_se=True)
    np.testing.assert_array_equal(
        np.asarray(iterms_pred, dtype=np.float64),
        np.asarray(terms_pred, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(iterms_se, dtype=np.float64),
        np.asarray(terms_se, dtype=np.float64),
    )

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _PREDICTION_FORMULA,
        family=_PREDICTION_FAMILY,
        method="ML",
        type="iterms",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(iterms_pred, dtype=np.float64),
        np.asarray(expected["pred"], dtype=np.float64),
        atol=_PREDICTION_ATOL,
        rtol=_PREDICTION_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(iterms_se, dtype=np.float64),
        np.asarray(expected["se"], dtype=np.float64),
        atol=_PREDICTION_ATOL,
        rtol=_PREDICTION_ATOL,
    )
