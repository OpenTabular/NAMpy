from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.parity.snapshots import _normalize_reference_term_label
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

pytestmark = [pytest.mark.surface_output]


def test_reference_term_label_normalization_drops_mgcv_constructor_options():
    """Verify that mgcv-style diagnostic labels omit constructor-only options."""
    assert (
        _normalize_reference_term_label(
            'ti(x0, x1, bs=["cr","ps"], k=[6,6], m=[2,3], mc=[TRUE,FALSE], fx=TRUE, xt=list(bs="ps"), sp=[1.0,1.2])'
        )
        == "ti(x0, x1)"
    )


def test_reference_term_label_normalization_preserves_factor_by_level_suffix():
    """Verify that factor-by smooth labels retain mgcv level identity."""
    assert (
        _normalize_reference_term_label(
            'te(x0, x1, by=f, bs=["cr","cr"], k=[5,5], sp=[1.0,1.2]):f=a'
        )
        == "te(x0, x1):fa"
    )


def test_concurvity_surfaces_match_mgcv_snapshot():
    """Verify that concurvity surfaces match mgcv snapshot."""
    case = get_parity_case("gaussian_cr_uni_reml")
    data = make_parity_case_data(case.case_id)

    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    gam = _fit_nampy_model(data, case.formula, case.family, case.method)

    actual_full = gam.concurvity(full=True)
    actual_pairwise = gam.concurvity(full=False)
    expected_diag = expected["parity"]["diagnostics"]

    assert [
        _normalize_reference_term_label(v) for v in actual_full["labels"]
    ] == expected_diag["concurvity_labels"]
    np.testing.assert_allclose(
        np.asarray(actual_full["values"], dtype=np.float64),
        np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )

    assert [
        _normalize_reference_term_label(v) for v in actual_pairwise["labels"]
    ] == expected_diag["concurvity_pairwise"]["labels"]
    for name in actual_pairwise["measure_names"]:
        np.testing.assert_allclose(
            np.asarray(actual_pairwise["values"][name], dtype=np.float64),
            np.asarray(expected_diag["concurvity_pairwise"][name], dtype=np.float64),
            atol=1e-8,
            rtol=0.0,
        )


def test_sp_vcov_and_one_se_rule_match_mgcv_snapshot():
    """Verify that sp vcov and one-standard-error rule match mgcv snapshot."""
    case = get_parity_case("poisson_cr_uni_reml")
    data = make_parity_case_data(case.case_id)

    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    gam = _fit_nampy_model(data, case.formula, case.family, case.method)
    expected_diag = expected["parity"]["diagnostics"]

    np.testing.assert_allclose(
        np.asarray(gam.sp_vcov(edge_correct=False), dtype=np.float64),
        np.asarray(expected_diag["sp_vcov"], dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(gam.one_se_rule(), dtype=np.float64),
        np.asarray(expected_diag["one_se_rule"], dtype=np.float64),
        atol=1e-5,
        rtol=2e-7,
    )


def test_gaussian_sp_vcov_and_one_se_rule_match_mgcv_snapshot():
    """Verify that gaussian sp vcov and one-standard-error rule match mgcv snapshot."""
    case = get_parity_case("gaussian_cr_uni_reml")
    data = make_parity_case_data(case.case_id)

    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    gam = _fit_nampy_model(data, case.formula, case.family, case.method)
    expected_diag = expected["parity"]["diagnostics"]

    np.testing.assert_allclose(
        np.asarray(gam.sp_vcov(edge_correct=False), dtype=np.float64),
        np.asarray(expected_diag["sp_vcov"], dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(gam.one_se_rule(), dtype=np.float64),
        np.asarray(expected_diag["one_se_rule"], dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(gam.one_se_rule(), dtype=np.float64),
        np.asarray(expected_diag["one_se_rule"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )
