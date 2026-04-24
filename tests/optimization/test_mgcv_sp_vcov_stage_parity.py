from __future__ import annotations

import numpy as np
import pytest

from tests.families.test_general_family_mgcv_parity import (
    GENERAL_TWO_CR_CASES,
    _gaulss_two_smooth_data,
    _shashlss_two_smooth_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _run_mgcv_gam_vcomp,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)
from tests.optimization.test_mgcv_vcomp_parity import _assert_gam_vcomp_close

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]

_ORDINARY_STAGE_CASES = [
    pytest.param("gaussian_cr_uni_reml", 1e-12, id="gaussian_cr_uni_reml"),
    pytest.param("poisson_cr_uni_reml", 3e-5, id="poisson_cr_uni_reml"),
]

_ONE_SE_STAGE_CASES = [
    pytest.param(
        "poisson_cr_uni_reml",
        None,
        None,
        None,
        None,
        1e-5,
        id="poisson_cr_uni_reml",
    ),
    pytest.param(
        "gaulss_two_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gaulss_two_smooth_data,
        "ML",
        1e-4,
        id="gaulss_two_cr",
    ),
]
_GAM_VCOMP_STAGE_CASES = [
    pytest.param(
        "poisson_cr_uni_reml",
        None,
        None,
        None,
        None,
        False,
        2e-5,
        id="poisson_cr_uni_reml_rescale_false",
    ),
    pytest.param(
        "shashlss_two_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_two_smooth_data,
        "ML",
        False,
        2e-5,
        id="shashlss_two_cr_rescale_false",
    ),
]
_UNCONDITIONAL_STAGE_CASES = [
    pytest.param(
        case_id,
        family,
        formula,
        data_factory,
        method,
        max(2e-5, float(pred_atol)),
        id=f"{case_id}_unconditional",
    )
    for (
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        _se_atol,
        _check_response_se,
    ) in GENERAL_TWO_CR_CASES
]

_GENERAL_GAP_REASONS = {
    "gevlss_two_cr": "gevlss sp.vcov matrix still differs from mgcv on the public parameterization.",
    "ziplss_two_cr": "ziplss sp.vcov matrix still differs from mgcv on the public parameterization.",
}

_GENERAL_STAGE_CASES = [
    pytest.param(
        case_id,
        family,
        formula,
        data_factory,
        method,
        max(1e-4, float(pred_atol)),
        id=case_id,
        marks=(
            [
                pytest.mark.status_known_gap,
                pytest.mark.xfail(strict=True, reason=_GENERAL_GAP_REASONS[case_id]),
            ]
            if case_id in _GENERAL_GAP_REASONS
            else []
        ),
    )
    for (
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        _se_atol,
        _check_response_se,
    ) in GENERAL_TWO_CR_CASES
]


def _assert_sp_vcov_close(data, formula, family, method, *, atol: float):
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)
    expected_diag = expected["parity"]["diagnostics"]

    np.testing.assert_allclose(
        np.asarray(gam.sp_vcov(edge_correct=False), dtype=np.float64),
        np.asarray(expected_diag["sp_vcov"], dtype=np.float64),
        atol=float(atol),
        rtol=0.0,
    )


def _assert_one_se_rule_close(data, formula, family, method, *, atol: float):
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)

    np.testing.assert_allclose(
        np.asarray(gam.one_se_rule(), dtype=np.float64),
        np.asarray(expected["parity"]["diagnostics"]["one_se_rule"], dtype=np.float64),
        atol=float(atol),
        rtol=float(atol),
    )


@pytest.mark.parametrize(("case_id", "atol"), _ORDINARY_STAGE_CASES)
def test_sp_vcov_matches_mgcv_snapshot_on_ordinary_public_parameterizations(
    case_id,
    atol,
):
    """
    Verify that sp vcov matches mgcv snapshot on ordinary public parameterizations.
    """
    case = get_parity_case(case_id)
    data = make_parity_case_data(case.case_id)
    _assert_sp_vcov_close(data, case.formula, case.family, case.method, atol=atol)


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _GENERAL_STAGE_CASES,
)
def test_sp_vcov_matches_mgcv_snapshot_on_general_family_two_smooth_branches(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """
    Verify that sp vcov matches mgcv snapshot on general family two smooth branches.
    """
    del case_id
    _assert_sp_vcov_close(data_factory(), formula, family, method, atol=atol)


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _ONE_SE_STAGE_CASES,
)
def test_one_se_rule_matches_mgcv_snapshot_on_representative_stage_cases(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """
    Verify that one-standard-error rule matches mgcv snapshot on representative stage
    cases.
    """
    if family is None:
        case = get_parity_case(case_id)
        data = make_parity_case_data(case.case_id)
        _assert_one_se_rule_close(
            data,
            case.formula,
            case.family,
            case.method,
            atol=atol,
        )
        return

    _assert_one_se_rule_close(data_factory(), formula, family, method, atol=atol)


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _UNCONDITIONAL_STAGE_CASES,
)
def test_unconditional_covariance_matches_mgcv_snapshot_on_general_family_stage_cases(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """
    Verify that unconditional covariance matches mgcv snapshot on general family stage
    cases.
    """
    del case_id
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)

    expected_cov = np.asarray(expected["fit"]["cov_unconditional"], dtype=np.float64)
    actual_fit_cov = np.asarray(
        gam.fit_core_solution_.fit_result.cov_unconditional, dtype=np.float64
    )
    actual_public_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)

    np.testing.assert_allclose(actual_fit_cov, expected_cov, atol=atol, rtol=0.0)
    np.testing.assert_allclose(actual_public_cov, expected_cov, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "rescale", "atol"),
    _GAM_VCOMP_STAGE_CASES,
)
def test_gam_vcomp_rescale_false_matches_mgcv_on_stage_cases(
    case_id,
    family,
    formula,
    data_factory,
    method,
    rescale,
    atol,
):
    """Verify that gam vcomp rescale false matches mgcv on stage cases."""
    if family is None:
        case = get_parity_case(case_id)
        data = make_parity_case_data(case.case_id)
        formula = case.formula
        family = case.family
        method = case.method
    else:
        data = data_factory()

    expected = _run_mgcv_gam_vcomp(data, formula, family, method, rescale=rescale)
    gam = _fit_nampy_model(data, formula, family, method)
    actual = gam.gam_vcomp(rescale=rescale)

    _assert_gam_vcomp_close(actual, expected, atol=atol)
