from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.parity.snapshots import _normalize_mgcv_term_label
from tests.families.test_general_family_mgcv_parity import (
    GENERAL_TWO_CR_CASES,
    _gaulss_data,
    _gaulss_two_smooth_data,
    _gevlss_data,
    _shashlss_data,
    _shashlss_two_smooth_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_gamma_data,
    _make_poisson_data,
    _run_mgcv_anova,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _normalize_numeric_matrix(x) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]

    def _coerce(value):
        if value is None or value == "NA":
            return np.nan
        return float(value)

    return np.vectorize(_coerce, otypes=[np.float64])(arr)

_ANOVA_CASES = [
    pytest.param(
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gaulss_data,
        1e-7,
        1e-7,
        slice(None),
        id="gaulss_cr",
    ),
    pytest.param(
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        8e-1,
        2e-2,
        slice(0, 3),
        id="gevlss_cr",
    ),
    pytest.param(
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        1e-2,
        1e-2,
        slice(None),
        id="shashlss_cr",
    ),
]

_UNCONDITIONAL_CASES = [
    pytest.param(
        "poisson_cr_uni_reml",
        None,
        None,
        None,
        None,
        2e-5,
        id="poisson_cr_uni_reml",
    ),
    pytest.param(
        "gaulss_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gaulss_data,
        "ML",
        2e-5,
        id="gaulss_cr",
    ),
]
_GENERAL_ANOVA_STAGE_CASES = [
    pytest.param(
        case_id,
        family,
        formula,
        data_factory,
        method,
        max(1e-2, float(pred_atol) * 100.0),
        max(1e-2, float(pred_atol) * 100.0),
        slice(None),
        id=case_id,
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
_GENERAL_UNCONDITIONAL_CASES = [
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

_ANOVA_COMPARISON_STAGE_CASES = [
    pytest.param(
        _make_poisson_data,
        "poisson",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
        2e-8,
        5e-6,
        id="poisson_two_cr",
    ),
    pytest.param(
        _make_gamma_data,
        "gamma",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
        1e-10,
        5e-6,
        id="gamma_two_cr",
    ),
]
_GENERAL_ANOVA_COMPARISON_STAGE_CASES = [
    pytest.param(
        _gaulss_two_smooth_data,
        "gaulss",
        [
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        ],
        "ML",
        2e-8,
        5e-6,
        id="gaulss_two_cr",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "General-family model-comparison anova still diverges from "
                    "mgcv on the Resid. Dev / Deviance summary surface."
                ),
            ),
        ],
    ),
    pytest.param(
        _shashlss_two_smooth_data,
        "shashlss",
        [
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            [
                'y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)',
                "~ 1",
                "~ 1",
                "~ 1",
            ],
        ],
        "ML",
        1e-6,
        5e-5,
        id="shashlss_two_cr",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "General-family model-comparison anova still diverges from "
                    "mgcv on the Resid. Dev / Deviance summary surface."
                ),
            ),
        ],
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
    ("family", "formula", "data_factory", "atol", "rtol", "compare_cols"),
    _ANOVA_CASES,
)
def test_general_family_single_model_anova_matches_mgcv_snapshot(
    family,
    formula,
    data_factory,
    atol,
    rtol,
    compare_cols,
):
    """Verify that general family single model anova matches mgcv snapshot."""
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    actual = gam.anova(freq=False)
    expected_block = expected["parity"]["diagnostics"]["anova_smooth"]
    expected_labels = [
        _normalize_mgcv_term_label(v)
        for v in np.atleast_1d(expected_block["labels"]).tolist()
    ]
    actual_labels = [
        _normalize_mgcv_term_label(v) for v in actual.smooth_table["label"].tolist()
    ]
    assert actual_labels == expected_labels

    actual_values = np.asarray(
        actual.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(),
        dtype=np.float64,
    )
    expected_values = np.atleast_2d(
        np.asarray(expected_block["values"], dtype=np.float64)
    )
    np.testing.assert_allclose(
        actual_values[:, compare_cols],
        expected_values[:, compare_cols],
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol", "rtol", "compare_cols"),
    _GENERAL_ANOVA_STAGE_CASES,
)
def test_general_family_multi_smooth_anova_matches_mgcv_snapshot(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
    rtol,
    compare_cols,
):
    """Verify that general family multi smooth anova matches mgcv snapshot."""
    del case_id
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)

    actual = gam.anova(freq=False)
    expected_block = expected["parity"]["diagnostics"]["anova_smooth"]
    expected_labels = [
        _normalize_mgcv_term_label(v)
        for v in np.atleast_1d(expected_block["labels"]).tolist()
    ]
    actual_labels = [
        _normalize_mgcv_term_label(v) for v in actual.smooth_table["label"].tolist()
    ]
    assert actual_labels == expected_labels

    actual_values = np.asarray(
        actual.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(),
        dtype=np.float64,
    )
    expected_values = np.atleast_2d(
        np.asarray(expected_block["values"], dtype=np.float64)
    )
    np.testing.assert_allclose(
        actual_values[:, compare_cols],
        expected_values[:, compare_cols],
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _UNCONDITIONAL_CASES,
)
def test_public_unconditional_covariance_matches_mgcv_snapshot(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """Verify that public unconditional covariance matches mgcv snapshot."""
    data, formula, family, method = _load_case(case_id, family, formula, data_factory, method)
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)

    expected_cov = expected["fit"]["cov_unconditional"]
    assert expected_cov is not None

    actual_fit_cov = gam.fit_core_solution_.fit_result.cov_unconditional
    actual_public_cov = gam.vcov(unconditional=True)

    np.testing.assert_allclose(
        np.asarray(actual_fit_cov, dtype=np.float64),
        np.asarray(expected_cov, dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_public_cov, dtype=np.float64),
        np.asarray(expected_cov, dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    _GENERAL_UNCONDITIONAL_CASES,
)
def test_general_family_two_smooth_unconditional_covariance_matches_mgcv_snapshot(
    case_id,
    family,
    formula,
    data_factory,
    method,
    atol,
):
    """
    Verify that general family two smooth unconditional covariance matches mgcv
    snapshot.
    """
    del case_id
    data = data_factory()
    expected = _run_mgcv_snapshot(data, formula, family, method)
    gam = _fit_nampy_model(data, formula, family, method)

    expected_cov = np.asarray(expected["fit"]["cov_unconditional"], dtype=np.float64)
    actual_fit_cov = np.asarray(
        gam.fit_core_solution_.fit_result.cov_unconditional,
        dtype=np.float64,
    )
    actual_public_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)

    np.testing.assert_allclose(actual_fit_cov, expected_cov, atol=atol, rtol=0.0)
    np.testing.assert_allclose(actual_public_cov, expected_cov, atol=atol, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(actual_public_cov, dtype=np.float64),
        np.asarray(expected_cov, dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("data_factory", "family", "formulas", "method", "deviance_tol", "df_tol"),
    _ANOVA_COMPARISON_STAGE_CASES,
)
def test_model_comparison_anova_matches_mgcv_on_representative_stage_cases(
    data_factory,
    family,
    formulas,
    method,
    deviance_tol,
    df_tol,
):
    """Verify that model comparison anova matches mgcv on representative stage cases."""
    data = data_factory()
    py0 = _fit_nampy_model(data, formulas[0], family, method)
    py1 = _fit_nampy_model(data, formulas[1], family, method)
    actual = py0.anova(py1, test="Chisq")
    expected = _run_mgcv_anova(data, formulas, family, method, test="Chisq")

    expected_values = _normalize_numeric_matrix(expected["table"]["values"])
    np.testing.assert_allclose(
        actual.table["Resid. Df"].to_numpy(dtype=np.float64),
        expected_values[:, 0],
        atol=df_tol,
        rtol=df_tol,
    )
    np.testing.assert_allclose(
        actual.table["Resid. Dev"].to_numpy(dtype=np.float64),
        expected_values[:, 1],
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual.table["Df"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 2]], dtype=np.float64),
        atol=df_tol,
        rtol=df_tol,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.table["Deviance"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 3]], dtype=np.float64),
        atol=deviance_tol,
        rtol=deviance_tol,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.table["Pr(>Chi)"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 4]], dtype=np.float64),
        atol=1e-12,
        rtol=1e-8,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("data_factory", "family", "formulas", "method", "deviance_tol", "df_tol"),
    _GENERAL_ANOVA_COMPARISON_STAGE_CASES,
)
def test_general_family_model_comparison_anova_matches_mgcv(
    data_factory,
    family,
    formulas,
    method,
    deviance_tol,
    df_tol,
):
    """Verify that general family model comparison anova matches mgcv."""
    data = data_factory()
    py0 = _fit_nampy_model(data, formulas[0], family, method)
    py1 = _fit_nampy_model(data, formulas[1], family, method)
    actual = py0.anova(py1, test="Chisq")
    expected = _run_mgcv_anova(data, formulas, family, method, test="Chisq")

    expected_values = _normalize_numeric_matrix(expected["table"]["values"])
    np.testing.assert_allclose(
        actual.table["Resid. Df"].to_numpy(dtype=np.float64),
        expected_values[:, 0],
        atol=df_tol,
        rtol=df_tol,
    )
    np.testing.assert_allclose(
        actual.table["Resid. Dev"].to_numpy(dtype=np.float64),
        expected_values[:, 1],
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual.table["Df"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 2]], dtype=np.float64),
        atol=df_tol,
        rtol=df_tol,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.table["Deviance"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 3]], dtype=np.float64),
        atol=deviance_tol,
        rtol=deviance_tol,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.table["Pr(>Chi)"].to_numpy(dtype=np.float64),
        np.asarray([np.nan, expected_values[1, 4]], dtype=np.float64),
        atol=1e-12,
        rtol=1e-8,
        equal_nan=True,
    )
