from __future__ import annotations

import numpy as np
import pytest

from nampy.gam._model_state import _term_blocks_seq
from nampy.gam.diagnostics.summary import summary_text
from tests.families.test_general_family_mgcv_parity import (
    _gammals_data,
    _gaulss_data,
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
    pytest.param(
        "poisson_cr_uni_reml", None, None, None, None, 1e-5, id="poisson_cr_uni_reml"
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
    data, formula, family, method = _load_case(
        case_id, family, formula, data_factory, method
    )
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
    data, formula, family, method = _load_case(
        case_id, family, formula, data_factory, method
    )
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
        snapshot_key = resid_type.replace(".", "_")
        expected_values = expected_residuals[snapshot_key]
        if expected_values is None:
            with pytest.raises(ValueError):
                gam.residuals(type=resid_type)
            continue

        actual = np.asarray(gam.residuals(type=resid_type), dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            np.asarray(expected_values, dtype=np.float64),
            atol=atol,
            rtol=atol,
        )


def test_general_family_summary_text_reports_public_term_labels_and_fit_scalars():
    """
    Verify that general family summary text reports public term labels and fit scalars.
    """
    data = _gaulss_data(seed=43)
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "ML")
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    text = summary_text(gam)
    assert f"Family : {gam.family.name}" in text
    assert f"Link : {gam.family.link_name}" in text
    assert f"Smoothing method : {gam._optim_method}" in text
    assert f"EDF (total) : {float(expected['fit']['edf_total']):.3f}" in text
    assert f"Scale : {float(expected['fit']['scale']):.6g}" in text
    assert f"Deviance : {float(expected['fit']['deviance']):.6g}" in text
    for tb in _term_blocks_seq(gam):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        assert tb.label in text
