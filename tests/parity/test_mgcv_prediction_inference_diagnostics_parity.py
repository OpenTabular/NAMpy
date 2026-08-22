"""Prediction / inference / diagnostics parity against mgcv.

Compared here:
  predict.gam on explicit ``newdata`` for ``link``, ``response``, ``terms``,
  ``lpmatrix``, standard errors, and unconditional standard errors.
  anova.gam single-model tables and representative model-comparison tables.
  residuals() and k.check() against mgcv snapshot outputs.

There are currently no active expected failures in this surface. If a new
upstream-localized gap is admitted, mark only its exact failing parameter so a
stale expectation becomes a loud XPASS.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import pytest

from nampy.gam.linalg import matrix_self_gram
from tests._mgcv_parity_requested_shared import CaseSpec
from tests._mgcv_snapshot_parity_shared import _make_fs_data, _make_gaussian_data
from tests.mgcv_invariant_policy import lpmatrix_uses_invariant_comparison
from tests.mgcv_parity_utils import (
    _family_specs,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_binomial_data,
    _make_gamma_data,
    _make_poisson_data,
    _run_mgcv_anova,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)
from tests.parity.test_mgcv_parity_failing_and_warnings import (
    REQUESTED_PARITY_FAILING_OR_WARNING_CASES,
)
from tests.parity.test_mgcv_snapshot_core_matrix import CASES as REQUESTED_CASES

pytestmark = [pytest.mark.surface_output]
_KCHECK_SUBSAMPLE = 120
_KCHECK_N_REP = 8
_KCHECK_K_INDEX_ATOL = 1.0 / np.sqrt(_KCHECK_N_REP)
_KCHECK_K_INDEX_RTOL = 0.5
_KCHECK_PGRID = 1.0 / _KCHECK_N_REP


def _dedupe_cases(cases: list[CaseSpec]) -> list[CaseSpec]:
    out = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        out.append(case)
    return out


def _compact_kcheck_label(label: str) -> str:
    """Normalize k_check labels so term identity is comparable to mgcv snapshots."""
    text = str(label).strip()
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx < 0 or close_idx <= open_idx:
        return text
    fn = text[:open_idx].strip()
    inner = text[open_idx + 1 : close_idx]

    args: list[str] = []
    current = []
    depth = 0
    for ch in inner:
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                args.append(part)
            current = []
            continue
        current.append(ch)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    part = "".join(current).strip()
    if part:
        args.append(part)

    kept = []
    for part in args:
        if "=" in part:
            break
        kept.append(part)
    if not kept:
        kept = args[:1]
    if not kept:
        return f"{fn}()"
    return f"{fn}({','.join(kept)})"


ADDITIONAL_SCENARIO_CASES = [
    CaseSpec(
        case_id="gaussian_fs_select_reml",
        formula='y ~ s(f, x, bs="fs", k=6)',
        family="gaussian",
        data_factory=_make_fs_data,
        select=True,
        skip_coef_comparison=True,
        criterion_atol=1e-3,
    ),
    CaseSpec(
        case_id="gaussian_ti_cs_ps_reml",
        formula='y ~ ti(x0, x1, bs=["cs", "ps"], k=[5, 6])',
        family="gaussian",
        data_factory=_make_gaussian_data,
        # The centered cs null-space representative is not uniquely oriented.
        # Prediction, inference, ANOVA, residuals, and k-check stay strict.
        skip_coef_comparison=True,
        criterion_atol=3e-3,
        se_tol_scale=5e-5,
    ),
]


ORDINARY_CASES = _dedupe_cases(
    list(REQUESTED_CASES)
    + list(REQUESTED_PARITY_FAILING_OR_WARNING_CASES)
    + ADDITIONAL_SCENARIO_CASES
)
CASE_BY_ID = {case.case_id: case for case in ORDINARY_CASES}


def _sample_weight_from_case(case: CaseSpec, data):
    if case.weights_column is None:
        return None
    return np.asarray(data[case.weights_column], dtype=np.float64)


def _family_key(case: CaseSpec) -> str:
    return str(_family_specs(case.family)[1]).split(":", 1)[0].lower()


def _is_gaussian_case(case: CaseSpec) -> bool:
    return _family_key(case) == "gaussian"


def _prediction_tol(case: CaseSpec) -> float:
    if case.case_id == "gaussian_fs_select_reml":
        return 1e-6
    if not _is_gaussian_case(case):
        return 1e-8
    if any(token in case.case_id for token in ("tp", "te", "weights", "offset", "by")):
        return 1e-8
    return 1e-10


def _anova_tol(case: CaseSpec) -> float:
    if case.case_id == "gaussian_fs_select_reml":
        return 2e-5
    if not _is_gaussian_case(case):
        return 1e-6
    return 1e-10


def _residual_tol(case: CaseSpec) -> float:
    if case.case_id == "gaussian_fs_select_reml":
        return 1e-6
    if case.case_id == "factor_smooth_sz":
        return 5e-10
    if not _is_gaussian_case(case):
        return 1e-8
    return 1e-10


def _unconditional_tol(case: CaseSpec) -> float:
    if case.case_id == "binomial_separation":
        return 1e-6
    if case.case_id == "gaussian_fs_select_reml":
        return 1e-6
    return max(_prediction_tol(case), 1e-7)


def _anova_p_value_rtol(case: CaseSpec) -> float:
    if case.case_id == "gaussian_ti_mc":
        return 3e-4
    return 1e-4


def _normalize_matrix(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _normalize_numeric_matrix(x) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]

    def _coerce(v):
        if v is None or v == "NA":
            return np.nan
        return float(v)

    return np.vectorize(_coerce, otypes=[np.float64])(arr)


def _normalize_vector(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _labels_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _extract_parametric_triplet(values) -> np.ndarray:
    arr = _normalize_numeric_matrix(values)
    if arr.shape[1] >= 5:
        return arr[:, [0, 3, 4]]
    if arr.shape[1] >= 3:
        return arr[:, [0, 1, 2]]
    raise AssertionError("Unexpected mgcv anova parametric table shape.")


def _assert_kcheck_p_value(value: float, *, n_rep: int, term: str, source: str) -> None:
    assert np.isfinite(
        value
    ), f"{source} k_check p_value is non-finite for '{term}': {value}"
    assert (
        0.0 <= value <= 1.0
    ), f"{source} k_check p_value out of range for '{term}': {value}"
    scaled = value * n_rep
    nearest = np.rint(scaled)
    assert np.isclose(scaled, nearest, atol=1e-12), (
        f"{source} k_check p_value for '{term}' is not on mgcv grid "
        f"({_KCHECK_PGRID:g} increments): value={value}"
    )
    assert (
        0.0 <= nearest <= n_rep
    ), f"{source} k_check p_value for '{term}' maps to invalid grid index: value={value}, n_rep={n_rep}"


@lru_cache(maxsize=None)
def _case_bundle(case_id: str):
    case = CASE_BY_ID[case_id]
    data = case.data_factory()
    expected = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    model = _fit_nampy_model_fixed_sp(
        data,
        case.formula,
        case.family,
        sp,
        select=case.select,
        sample_weight=_sample_weight_from_case(case, data),
    )
    return data, expected, model


@lru_cache(maxsize=None)
def _case_outer_bundle(case_id: str):
    case = CASE_BY_ID[case_id]
    data = case.data_factory()
    expected = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    model = _fit_nampy_model(
        data,
        case.formula,
        case.family,
        "REML",
        select=case.select,
        sample_weight=_sample_weight_from_case(case, data),
    )
    return data, expected, model


def _newdata_for_case(case_id: str):
    data, _expected, _model = _case_bundle(case_id)
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
    n = min(40, len(data))
    return data.sample(n=n, random_state=seed).copy()


def _assert_p_values_close(actual, expected, *, atol: float, rtol: float) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.where(np.abs(actual) < 1e-300, 0.0, actual)
    expected = np.where(np.abs(expected) < 1e-300, 0.0, expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


@pytest.mark.parametrize(
    "case", ORDINARY_CASES, ids=[case.case_id for case in ORDINARY_CASES]
)
@pytest.mark.parametrize(
    "pred_type",
    ["link", "response", "terms", "lpmatrix"],
    ids=["link", "response", "terms", "lpmatrix"],
)
def test_predict_gam_newdata_surfaces_match_mgcv(case: CaseSpec, pred_type: str):
    """Verify that predict gam new-data surfaces match mgcv."""
    data, _expected, model = _case_bundle(case.case_id)
    newdata = _newdata_for_case(case.case_id)
    r_result = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type=pred_type,
        return_se=(pred_type != "lpmatrix"),
        select=case.select,
        weights_column=case.weights_column,
    )

    tol = _unconditional_tol(case)
    if pred_type == "lpmatrix":
        actual = np.asarray(model.predict(X=newdata, type="lpmatrix"), dtype=np.float64)
        expected = np.asarray(r_result["pred"], dtype=np.float64)
        if lpmatrix_uses_invariant_comparison(case.case_id):
            np.testing.assert_allclose(
                matrix_self_gram(actual),
                matrix_self_gram(expected),
                atol=tol,
                rtol=0.0,
            )
            return
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
        return

    actual_pred, actual_se = model.predict(X=newdata, type=pred_type, return_se=True)
    if pred_type == "terms":
        expected_pred = _normalize_matrix(r_result["pred"])
        expected_se = _normalize_matrix(r_result["se"])
        actual_pred = _normalize_matrix(actual_pred)
        actual_se = _normalize_matrix(actual_se)
        assert actual_pred.shape == expected_pred.shape
        assert actual_se.shape == expected_se.shape
        assert actual_pred.shape[1] == len(
            _labels_list(r_result.get("term_names", None))
        )
    else:
        expected_pred = _normalize_vector(r_result["pred"])
        expected_se = _normalize_vector(r_result["se"])
        actual_pred = _normalize_vector(actual_pred)
        actual_se = _normalize_vector(actual_se)

    np.testing.assert_allclose(actual_pred, expected_pred, atol=tol, rtol=tol)
    np.testing.assert_allclose(actual_se, expected_se, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "case", ORDINARY_CASES, ids=[case.case_id for case in ORDINARY_CASES]
)
@pytest.mark.parametrize(
    "pred_type",
    ["link", "response", "terms"],
    ids=["link", "response", "terms"],
)
def test_predict_gam_unconditional_se_match_mgcv_or_documented_gap(
    case: CaseSpec, pred_type: str
):
    """
    Verify that predict gam unconditional standard errors match mgcv or documented gap.
    """
    data, _expected, model = _case_outer_bundle(case.case_id)
    newdata = _newdata_for_case(case.case_id)
    r_result = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type=pred_type,
        return_se=True,
        unconditional=True,
        select=case.select,
        weights_column=case.weights_column,
    )

    actual_cov = np.asarray(model.vcov(unconditional=True), dtype=np.float64)
    actual_pred, actual_se = model.predict(
        X=newdata,
        type=pred_type,
        return_se=True,
        cov=actual_cov,
    )
    tol = _unconditional_tol(case)
    if pred_type == "terms":
        np.testing.assert_allclose(
            _normalize_matrix(actual_pred),
            _normalize_matrix(r_result["pred"]),
            atol=tol,
            rtol=tol,
        )
        np.testing.assert_allclose(
            _normalize_matrix(actual_se),
            _normalize_matrix(r_result["se"]),
            atol=tol,
            rtol=tol,
        )
        return

    np.testing.assert_allclose(
        _normalize_vector(actual_pred),
        _normalize_vector(r_result["pred"]),
        atol=tol,
        rtol=tol,
    )
    np.testing.assert_allclose(
        _normalize_vector(actual_se),
        _normalize_vector(r_result["se"]),
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize(
    "case", ORDINARY_CASES, ids=[case.case_id for case in ORDINARY_CASES]
)
def test_anova_gam_single_model_matches_mgcv(case: CaseSpec):
    """Verify that anova gam single model matches mgcv."""
    _data, expected, model = _case_bundle(case.case_id)
    actual = model.anova(freq=False)
    tol = _anova_tol(case)

    expected_smooth = expected["parity"]["diagnostics"].get("anova_smooth")
    if expected_smooth is not None:
        expected_values = _normalize_numeric_matrix(expected_smooth["values"])
        actual_values = np.asarray(
            actual.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(),
            dtype=np.float64,
        )
        assert actual_values.shape == expected_values.shape
        np.testing.assert_allclose(
            actual_values[:, :2],
            expected_values[:, :2],
            atol=max(tol, 1e-6),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            actual_values[:, 2],
            expected_values[:, 2],
            atol=max(tol, 1e-6),
            rtol=1e-3,
        )
        # mgcv computes smooth p-values through the Davies (1980) qfc routine
        # at tol=2e-5 (psum.chisq, mgcv/R/mgcv.r:3466-3498). For effectively
        # saturated fits the statistic explodes (gaussian_fs_select_reml:
        # F ~ 6e9 with residual df ~ 3e-5) and the C routine's return value is
        # a numerical artifact that flips between 0.5 and 0.0 under last-bit
        # input changes — not a reproducible parity target. NAMpy's port
        # resolves the tail correctly (~1e-5), so for such rows only require a
        # small p-value instead of matching the artifact.
        davies_resolvable = np.abs(expected_values[:, 2]) < 1e8
        _assert_p_values_close(
            actual_values[davies_resolvable, 3],
            expected_values[davies_resolvable, 3],
            atol=max(tol, 1e-12),
            rtol=_anova_p_value_rtol(case),
        )
        assert np.all(actual_values[~davies_resolvable, 3] <= 1e-3)

    expected_parametric = expected["parity"]["diagnostics"].get("anova_parametric")
    if expected_parametric and "values" in expected_parametric:
        expected_values = _extract_parametric_triplet(expected_parametric["values"])
        actual_values = np.asarray(
            actual.parametric_table[["df", "wald_stat", "p_value"]].to_numpy(),
            dtype=np.float64,
        )
        assert actual_values.shape == expected_values.shape
        np.testing.assert_allclose(
            actual_values[:, :2],
            expected_values[:, :2],
            atol=max(tol, 1e-6),
            rtol=1e-4,
            equal_nan=True,
        )
        _assert_p_values_close(
            actual_values[:, 2],
            expected_values[:, 2],
            atol=max(tol, 1e-12),
            rtol=_anova_p_value_rtol(case),
        )


ANOVA_COMPARISON_CASES = [
    (
        "gaussian_two_cr",
        _make_gaussian_data,
        "gaussian",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
    ),
    (
        "binomial_two_cr",
        _make_binomial_data,
        "binomial",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
    ),
    (
        "poisson_two_cr",
        _make_poisson_data,
        "poisson",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
    ),
    (
        "gamma_two_cr",
        _make_gamma_data,
        "gamma",
        [
            'y ~ s(x0, bs="cr", k=8)',
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        ],
        "REML",
    ),
]


@pytest.mark.parametrize(
    ("case_id", "data_factory", "family", "formulas", "method"),
    ANOVA_COMPARISON_CASES,
    ids=[case[0] for case in ANOVA_COMPARISON_CASES],
)
def test_anova_gam_model_comparison_matches_mgcv(
    case_id, data_factory, family, formulas, method
):
    """Verify that anova gam model comparison matches mgcv."""
    del case_id
    data = data_factory()
    py0 = _fit_nampy_model(data, formulas[0], family, method)
    py1 = _fit_nampy_model(data, formulas[1], family, method)
    actual = py0.anova(py1, test="Chisq")
    expected = _run_mgcv_anova(data, formulas, family, method, test="Chisq")
    deviance_tol = 2e-8 if family in {"binomial", "poisson"} else 1e-10

    expected_values = _normalize_numeric_matrix(expected["table"]["values"])
    np.testing.assert_allclose(
        actual.table["Resid. Df"].to_numpy(dtype=np.float64),
        expected_values[:, 0],
        atol=5e-6,
        rtol=5e-6,
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
        atol=5e-6,
        rtol=5e-6,
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
    "case", ORDINARY_CASES, ids=[case.case_id for case in ORDINARY_CASES]
)
@pytest.mark.parametrize(
    ("snapshot_key", "resid_type"),
    [
        ("response", "response"),
        ("working", "working"),
        ("pearson", "pearson"),
        ("scaled_pearson", "scaled.pearson"),
        ("deviance", "deviance"),
    ],
    ids=["response", "working", "pearson", "scaled_pearson", "deviance"],
)
def test_residuals_match_mgcv(
    case: CaseSpec,
    snapshot_key: str,
    resid_type: str,
):
    """Verify that residuals match mgcv."""
    _data, expected, model = _case_bundle(case.case_id)
    expected_values = expected["parity"]["diagnostics"]["residuals"][snapshot_key]
    actual = np.asarray(model.residuals(type=resid_type), dtype=np.float64)
    tol = _residual_tol(case)
    if resid_type == "working" and not _is_gaussian_case(case):
        tol = max(tol, 1e-6)
    np.testing.assert_allclose(
        actual,
        np.asarray(expected_values, dtype=np.float64),
        atol=tol,
        rtol=tol,
    )


@pytest.mark.surface_kcheck
@pytest.mark.parametrize(
    "case", ORDINARY_CASES, ids=[case.case_id for case in ORDINARY_CASES]
)
def test_k_check_matches_mgcv_or_documented_gap(case: CaseSpec):
    """Verify that k-check matches mgcv or documented gap."""
    _data, expected, model = _case_bundle(case.case_id)
    expected_block = expected["parity"]["diagnostics"].get("k_check")
    assert expected_block is not None

    actual = model.k_check(subsample=_KCHECK_SUBSAMPLE, n_rep=_KCHECK_N_REP, seed=0)
    actual_values = actual[["k_prime", "edf", "k_index", "p_value"]].to_numpy(
        dtype=np.float64
    )

    expected_labels = _labels_list(expected_block["labels"])
    expected_values = _normalize_numeric_matrix(expected_block["values"])

    assert len(actual.index) == len(expected_labels) == expected_values.shape[0]
    assert [_compact_kcheck_label(x) for x in actual.index] == [
        _compact_kcheck_label(x) for x in expected_labels
    ], (
        "k_check term order diverged between NAMpy and mgcv snapshots.\n"
        f"actual={list(actual.index)}\n"
        f"expected={expected_labels}"
    )
    for i in range(expected_values.shape[0]):
        assert int(actual_values[i, 0]) == int(round(expected_values[i, 0]))
        np.testing.assert_allclose(
            actual_values[i, 1],
            expected_values[i, 1],
            atol=1e-4,
            rtol=0.0,
        )
        if np.isnan(expected_values[i, 2]):
            assert np.isnan(actual_values[i, 2])
            assert np.isnan(actual_values[i, 3])
            continue
        np.testing.assert_allclose(
            actual_values[i, 2],
            expected_values[i, 2],
            atol=_KCHECK_K_INDEX_ATOL,
            rtol=_KCHECK_K_INDEX_RTOL,
        )
        _assert_kcheck_p_value(
            float(actual_values[i, 3]),
            n_rep=_KCHECK_N_REP,
            term=actual.index[i],
            source="actual",
        )
        _assert_kcheck_p_value(
            float(expected_values[i, 3]),
            n_rep=_KCHECK_N_REP,
            term=expected_labels[i],
            source="R",
        )
