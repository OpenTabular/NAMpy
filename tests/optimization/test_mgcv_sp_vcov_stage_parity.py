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


def test_sp_vcov_edge_correct_default_uses_hess1_and_matches_mgcv():
    """The default sp_vcov() path (edge_correct=True) mirrors mgcv exactly.

    Upstream mgcv/R/mgcv.r:4221-4233 (sp.vcov): with an edge-corrected fit,
    the covariance is solve(hess1 + diag(p)*reg) anchored at lsp1; without
    hess1 it falls back to solve(hess + reg) where the scalar reg is added to
    every element (R recycling) — an intentional asymmetry. Both branches are
    reproduced here from mgcv's own hess/hess1 payloads on the same
    edge-corrected fit, and the two results are asserted to genuinely differ
    so the branch selection itself is observable.
    """
    from tests.optimization.test_mgcv_outer_optimization_parity import (
        _finalize_python_edge_correct_fit,
        _run_mgcv_outer_trace,
    )

    data = make_parity_case_data("poisson_cr_uni_reml")
    spec = get_parity_case("poisson_cr_uni_reml")
    expected = _run_mgcv_outer_trace(
        data, spec.formula, "poisson", "REML", "newton", edge_correct=True
    )
    gam, _result = _finalize_python_edge_correct_fit(
        data, spec.formula, "poisson"
    )

    outer = expected["fit"]["outer_info"]
    hess1_r = np.asarray(outer["hess1"], dtype=np.float64)
    hess_r = np.asarray(outer["hessian"], dtype=np.float64)
    p = hess1_r.shape[0]
    reg = 1e-3
    v_expected_edge = np.linalg.solve(
        hess1_r + np.eye(p) * reg, np.eye(p)
    )
    v_expected_plain = np.linalg.solve(hess_r + reg, np.eye(p))

    v_edge = np.asarray(gam.sp_vcov(), dtype=np.float64)
    v_plain = np.asarray(gam.sp_vcov(edge_correct=False), dtype=np.float64)

    np.testing.assert_allclose(v_edge, v_expected_edge, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(v_plain, v_expected_plain, atol=1e-5, rtol=1e-5)
    assert not np.allclose(v_edge, v_plain, atol=1e-8, rtol=1e-8)
