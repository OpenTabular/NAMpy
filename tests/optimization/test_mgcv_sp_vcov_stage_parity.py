from __future__ import annotations

import numpy as np
import pytest

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
