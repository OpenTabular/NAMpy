from __future__ import annotations

import numpy as np
import pytest

from nampy.gam._model_state import _term_blocks_seq
from nampy.gam.diagnostics.summary import summary_text
from nampy.gam.parity.snapshots import _normalize_mgcv_term_label
from tests.diagnostics.test_mgcv_k_check_parity import (
    _assert_k_check_parity,
    _nampy_k_check,
    _r_k_check,
)
from tests.families.test_general_family_mgcv_parity import (
    GENERAL_TWO_CR_CASES,
    _gammals_data,
    _gaulss_data,
    _gaulss_two_smooth_data,
    _gevlss_data,
    _shashlss_data,
    _ziplss_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_gam_vcomp,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)
from tests.optimization.test_mgcv_vcomp_parity import _assert_gam_vcomp_close

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]

_ONE_SE_CASES = [
    pytest.param("poisson_cr_uni_reml", None, None, None, None, 1e-5, id="poisson_cr_uni_reml"),
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

_VCOMP_CASES = [
    pytest.param(
        "poisson_cr_uni_reml",
        None,
        None,
        None,
        None,
        False,
        2e-5,
        id="poisson_reml_rescale_false",
    ),
    pytest.param(
        "gaulss_two_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gaulss_two_smooth_data,
        "ML",
        False,
        2e-5,
        id="gaulss_two_cr_rescale_false",
    ),
]

_RESIDUAL_STAGE_CASES = [
    pytest.param(
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gaulss_data,
        ("response", "pearson", "deviance"),
        2e-7,
        id="gaulss_cr",
    ),
    pytest.param(
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        ("response", "pearson", "deviance"),
        2e-6,
        id="gevlss_cr",
    ),
    pytest.param(
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        ("response", "deviance"),
        2e-6,
        id="shashlss_cr",
    ),
    pytest.param(
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        ("response", "deviance"),
        2e-6,
        id="ziplss_cr",
    ),
]
_ADDITIONAL_RESIDUAL_STAGE_CASES = [
    pytest.param(
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gaulss_data,
        ("working", "scaled.pearson"),
        2e-7,
        id="gaulss_extra",
    ),
    pytest.param(
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        ("working", "pearson", "scaled.pearson"),
        2e-5,
        id="gammals_extra",
    ),
    pytest.param(
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        ("working", "scaled.pearson"),
        2e-6,
        id="gevlss_extra",
    ),
]


def _load_case(case_id, family, formula, data_factory, method):
    if family is None:
        case = get_parity_case(case_id)
        return (
            make_parity_case_data(case.case_id),
            case.formula,
            case.family,
            case.method,
        )
    return data_factory(), formula, family, method


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _ONE_SE_CASES,
)
def test_one_se_rule_matches_mgcv_snapshot_on_stage_cases(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """Verify that one-standard-error rule matches mgcv snapshot on stage cases."""
    data, formula, family, method = _load_case(case_id, family, formula, data_factory, method)
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)

    np.testing.assert_allclose(
        np.asarray(gam.one_se_rule(), dtype=np.float64),
        np.asarray(expected["parity"]["diagnostics"]["one_se_rule"], dtype=np.float64),
        atol=atol,
        rtol=atol,
    )


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "rescale", "atol"),
    _VCOMP_CASES,
)
def test_gam_vcomp_matches_mgcv_on_representative_stage_cases(
    case_id,
    family,
    formula,
    data_factory,
    method,
    rescale,
    atol,
):
    """Verify that gam vcomp matches mgcv on representative stage cases."""
    data, formula, family, method = _load_case(case_id, family, formula, data_factory, method)
    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        family,
        method,
        rescale=rescale,
    )
    gam = _fit_nampy_model(data, formula, family, method)
    actual = gam.gam_vcomp(rescale=rescale)

    _assert_gam_vcomp_close(actual, expected, atol=atol)


def test_general_family_concurvity_and_k_check_match_mgcv_snapshot():
    """Verify that general family concurvity and k-check match mgcv snapshot."""
    data = _gaulss_two_smooth_data(seed=41)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "ML")
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")
    expected_diag = expected["parity"]["diagnostics"]

    actual_full = gam.concurvity(full=True)
    actual_pairwise = gam.concurvity(full=False)

    assert [
        _normalize_mgcv_term_label(v) for v in actual_full["labels"]
    ] == expected_diag["concurvity_labels"]
    np.testing.assert_allclose(
        np.asarray(actual_full["values"], dtype=np.float64),
        np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
        atol=1e-4,
        rtol=0.0,
    )

    assert [
        _normalize_mgcv_term_label(v) for v in actual_pairwise["labels"]
    ] == expected_diag["concurvity_pairwise"]["labels"]
    for name in actual_pairwise["measure_names"]:
        np.testing.assert_allclose(
            np.asarray(actual_pairwise["values"][name], dtype=np.float64),
            np.asarray(expected_diag["concurvity_pairwise"][name], dtype=np.float64),
            atol=1e-4,
            rtol=0.0,
        )

    r_block = _r_k_check(expected)
    assert r_block is not None
    py_labels, py_values = _nampy_k_check(gam)
    _assert_k_check_parity(
        r_block,
        py_labels,
        py_values,
        numeric_terms={"x", "z"},
        edf_atol=1e-4,
    )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "residual_types", "atol"),
    _RESIDUAL_STAGE_CASES,
)
def test_general_family_residual_types_match_mgcv_snapshot(
    family,
    formula,
    data_factory,
    residual_types,
    atol,
):
    """Verify that general family residual types match mgcv snapshot."""
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    expected_residuals = expected["parity"]["diagnostics"]["residuals"]

    for resid_type in residual_types:
        actual = np.asarray(gam.residuals(type=resid_type), dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            np.asarray(expected_residuals[resid_type], dtype=np.float64),
            atol=atol,
            rtol=atol,
        )


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "residual_types", "atol"),
    _ADDITIONAL_RESIDUAL_STAGE_CASES,
)
def test_general_family_additional_residual_types_match_mgcv_snapshot(
    family,
    formula,
    data_factory,
    residual_types,
    atol,
):
    """Verify that general family additional residual types match mgcv snapshot."""
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    expected_residuals = expected["parity"]["diagnostics"]["residuals"]

    for resid_type in residual_types:
        actual = np.asarray(gam.residuals(type=resid_type), dtype=np.float64)
        snapshot_key = resid_type.replace(".", "_")
        np.testing.assert_allclose(
            actual,
            np.asarray(expected_residuals[snapshot_key], dtype=np.float64),
            atol=atol,
            rtol=atol,
        )


def test_general_family_summary_text_reports_public_term_labels_and_fit_scalars():
    """
    Verify that general family summary text reports public term labels and fit scalars.
    """
    data = _gaulss_two_smooth_data(seed=43)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    text = summary_text(gam)
    assert f"Family : {gam.family.name}" in text
    assert f"Link : {gam.family.link_name}" in text
    assert f"Smoothing method : {gam._optim_method}" in text
    for tb in _term_blocks_seq(gam):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        assert tb.label in text


@pytest.mark.parametrize(
    (
        "case_id",
        "family",
        "formula",
        "data_factory",
        "method",
        "pred_atol",
        "_se_atol",
        "_check_response_se",
    ),
    GENERAL_TWO_CR_CASES,
    ids=[case[0] for case in GENERAL_TWO_CR_CASES],
)
def test_general_family_smooth_function_space_matches_mgcv_snapshot(
    case_id,
    family,
    formula,
    data_factory,
    method,
    pred_atol,
    _se_atol,
    _check_response_se,
):
    """Verify that general family smooth function space matches mgcv snapshot."""
    del case_id, _se_atol, _check_response_se
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)
    actual = gam.parity_snapshot(X=data, include_covariances=True)["parity"][
        "diagnostics"
    ]["smooth_function_space"]
    expected_block = expected["parity"]["diagnostics"]["smooth_function_space"]

    assert actual["labels"] == expected_block["labels"]
    for got, want in zip(actual["fitted"], expected_block["fitted"]):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            np.asarray(want, dtype=np.float64),
            atol=max(1e-5, float(pred_atol)),
            rtol=max(1e-5, float(pred_atol)),
        )
    for got, want in zip(actual["variance_diag"], expected_block["variance_diag"]):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            np.asarray(want, dtype=np.float64),
            atol=max(1e-5, float(pred_atol)),
            rtol=max(1e-5, float(pred_atol)),
        )


def test_general_family_summary_text_scalars_match_mgcv_snapshot():
    """Verify that general family summary text scalars match mgcv snapshot."""
    data = _gaulss_two_smooth_data(seed=47)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "ML")
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    text = summary_text(gam)
    assert f"EDF (total) : {float(expected['fit']['edf_total']):.3f}" in text
    assert f"Scale : {float(expected['fit']['scale']):.6g}" in text
    assert f"Deviance : {float(expected['fit']['deviance']):.6g}" in text
