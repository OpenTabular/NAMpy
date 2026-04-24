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
    "gaulss_select_true_cr",
    "gammals_numeric_by",
    "gevlss_select_true_cr",
    "shashlss_numeric_by",
    "ziplss_numeric_by",
    "ziplss_select_true_cr",
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
_TENSOR_PUBLIC_GAP_CASES = [
    pytest.param(
        "gaulss_t2_full_false",
        id="gaulss_t2_full_false",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "gaulss_t2_full_true",
        id="gaulss_t2_full_true",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "gammals_t2_full_false",
        id="gammals_t2_full_false",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "gevlss_t2_full_false",
        id="gevlss_t2_full_false",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "shashlss_t2_full_false",
        id="shashlss_t2_full_false",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "ziplss_t2_full_false",
        id="ziplss_t2_full_false",
        marks=[pytest.mark.status_known_gap],
    ),
    pytest.param(
        "ziplss_t2_full_true",
        id="ziplss_t2_full_true",
        marks=[pytest.mark.status_known_gap],
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


@pytest.mark.parametrize("case_id", _TENSOR_PUBLIC_GAP_CASES)
@pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
def test_general_family_tensor_heavy_prediction_case_stays_localized_to_known_gap(
    case_id,
    pred_type,
):
    """
    Verify that general family tensor heavy prediction case stays localized to known
    gap.
    """
    family, formula, data_factory, method, pred_atol, se_atol, check_response_se = (
        _load_general_stage_case(case_id)
    )
    data = data_factory()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(data, formula, family, method)
    if pred_type == "terms":
        with pytest.raises(
            NotImplementedError,
            match="prediction parameterization is wider than the fitted coefficient space",
        ):
            gam.predict(newdata, type=pred_type, return_se=True)
        return
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


@pytest.mark.parametrize("iterms_type", [None, 2], ids=["default", "type_2"])
def test_general_family_iterms_rejects_multi_predictor_public_surface(iterms_type):
    """Verify that multi-predictor general-family iterms downgrades to terms like mgcv."""
    data = _gaulss_data(seed=17)
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(
        data,
        _PREDICTION_FORMULA,
        _PREDICTION_FAMILY,
        _PREDICTION_METHOD,
    )

    expected_pred, expected_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
    )

    with pytest.warns(
        UserWarning,
        match="type='iterms' not available for multiple predictor cases; using type='terms' instead.",
    ):
        actual_pred, actual_se = gam.predict(
            newdata,
            type="iterms",
            return_se=True,
            iterms_type=iterms_type,
        )

    _assert_general_prediction_close(actual_pred, expected_pred, atol=1e-7)
    _assert_general_prediction_close(actual_se, expected_se, atol=1e-7)
