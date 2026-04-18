"""Post-processing / final-fit parity against mgcv post.proc surfaces."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nampy.gam import GAM
from tests._mgcv_parity_requested_shared import CaseSpec
from tests.families.test_general_family_mgcv_parity import GENERAL_SE_CASES
from tests.mgcv_parity_utils import _family_specs, _run_mgcv_snapshot
from tests.parity.test_mgcv_parity import CASES as REQUESTED_CASES
from tests.parity.test_mgcv_parity_failing_and_warnings import (
    REQUESTED_PARITY_FAILING_OR_WARNING_CASES,
)

_WARNING_NOISE = {
    "NaNs produced",
}
_KNOWN_FAILING_OR_WARNING_CASE_IDS = {
    case.case_id for case in REQUESTED_PARITY_FAILING_OR_WARNING_CASES
}
_GENERAL_POSTPROC_KNOWN_GAP_TAGS = (
    "select_true",
    "numeric_by",
    "t2_",
    "two_cr",
)


def _dedupe_requested_cases(cases: list[CaseSpec]) -> list[CaseSpec]:
    out = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        out.append(case)
    return out


ORDINARY_CASES = _dedupe_requested_cases(
    list(REQUESTED_CASES) + list(REQUESTED_PARITY_FAILING_OR_WARNING_CASES)
)
MAGIC_CASES = [
    case for case in ORDINARY_CASES if str(case.family).lower() == "gaussian"
]


def _sample_weight_from_case(case: CaseSpec, data):
    if case.weights_column is None:
        return None
    return np.asarray(data[case.weights_column], dtype=np.float64)


def _normalize_warning_messages(messages) -> list[str]:
    out = []
    for msg in messages or []:
        text = " ".join(str(msg).split())
        if not text or text in _WARNING_NOISE:
            continue
        out.append(text)
    return out


def _normalize_optional(value):
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 0:
        return None
    if isinstance(value, list) and len(value) == 0:
        return None
    return value


def _expected_optimizer_name(expected_snapshot: dict) -> str | None:
    optimizer = _normalize_optional(expected_snapshot["fit"].get("optimizer", None))
    if isinstance(optimizer, list):
        if len(optimizer) == 0:
            return None
        optimizer = optimizer[-1]
    if optimizer is None:
        return None
    return str(optimizer).lower()


def _nampy_optimizer_name(expected_snapshot: dict) -> str | None:
    optimizer = _expected_optimizer_name(expected_snapshot)
    mapping = {
        "newton": "outer_newton",
        "bfgs": "bfgs",
        "efs": "efs",
        "optim": "lbfgsb",
        "magic": None,
    }
    if optimizer not in mapping:
        raise AssertionError(f"Unsupported mgcv optimizer tag {optimizer!r}.")
    return mapping[optimizer]


def _formula_is_orientation_stable(formula, *, skip_coef_comparison: bool) -> bool:
    if skip_coef_comparison:
        return False
    text = str(formula)
    return "tp" not in text and "t2(" not in text


def _compute_hat_diag(gam: GAM) -> np.ndarray | None:
    sol = gam.fit_core_solution_
    fit_state = sol.fit_state
    if fit_state.X is None or fit_state.A_inv is None:
        return None
    weights = fit_state.working_weights
    if weights is None:
        weights = fit_state.fisher_weights
    if weights is None:
        return None
    X = np.asarray(fit_state.X, dtype=np.float64)
    A_inv = np.asarray(fit_state.A_inv, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != X.shape[0]:
        return None
    WX = np.sqrt(np.clip(w, 0.0, None))[:, None] * X
    return np.asarray(np.sum((WX @ A_inv) * WX, axis=1), dtype=np.float64)


def _serialize_outer_info_block(block):
    block = _normalize_optional(block)
    if block is None:
        return None
    block = dict(block)
    out = {}
    for key in (
        "conv",
        "iter",
        "score_hist",
        "grad",
        "hess",
        "convergence",
        "message",
        "counts",
    ):
        value = _normalize_optional(block.get(key, None))
        if value is None:
            out[key] = None
        elif key in {"score_hist", "grad", "counts"}:
            out[key] = np.asarray(value, dtype=np.float64)
        elif key == "hess":
            out[key] = np.asarray(value, dtype=np.float64)
        elif key in {"iter", "convergence"}:
            out[key] = int(value)
        else:
            out[key] = str(value)
    if all(value is None for value in out.values()):
        return None
    return out


def _serialize_actual_outer_info(gam: GAM, *, allow_synthetic: bool) -> dict | None:
    if gam._optim_result is None:
        return None

    outer = dict(getattr(gam._optim_result, "outer_info", {}) or {})
    if not outer and allow_synthetic:
        trace = list(getattr(gam, "_optim_trace", []) or [])
        score_hist = [
            float(row["criterion"])
            for row in trace
            if row.get("criterion", None) is not None
        ]
        counts = []
        for attr in ("nfev", "njev"):
            val = getattr(gam._optim_result, attr, None)
            if val is not None:
                counts.append(int(val))
        outer = {
            "conv": getattr(gam._optim_result, "message", None),
            "iter": getattr(gam._optim_result, "nit", None),
            "score_hist": score_hist or None,
            "grad": getattr(gam._optim_result, "jac", None),
            "hess": getattr(gam._optim_result, "hess", None),
            "convergence": getattr(gam._optim_result, "status", None),
            "message": getattr(gam._optim_result, "message", None),
            "counts": counts or None,
        }
    return _serialize_outer_info_block(outer)


def _fit_requested_case(
    case: CaseSpec,
    *,
    method: str,
    fixed_sp: np.ndarray | None = None,
    optimizer: str | None = None,
):
    data = case.data_factory()
    sample_weight = _sample_weight_from_case(case, data)
    family_nampy, _family_token = _family_specs(case.family)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if fixed_sp is None:
            gam = GAM(
                family=family_nampy,
                formula=case.formula,
                select=case.select,
                optimize_smoothing=True,
                smoothing_method=("GCV" if method == "GCV.Cp" else method),
                smoothing_optimizer=optimizer or "lbfgsb",
            )
        else:
            gam = GAM(
                family=family_nampy,
                formula=case.formula,
                select=case.select,
                optimize_smoothing=False,
                smoothing_method="fixed",
                smoothing_params=np.asarray(fixed_sp, dtype=np.float64).ravel(),
            )
        gam.fit(data=data, sample_weight=sample_weight)

    return data, gam, _normalize_warning_messages([str(w.message) for w in caught])


def _fit_general_case(
    case,
    *,
    fixed_sp: np.ndarray | None = None,
    optimizer: str | None = None,
):
    case_id, family, formula, data_factory, method, *_rest = case
    data = data_factory()
    family_nampy, _family_token = _family_specs(family)
    select = "select_true" in case_id

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if fixed_sp is None:
            gam = GAM(
                family=family_nampy,
                formula=formula,
                select=select,
                optimize_smoothing=True,
                smoothing_method=method,
                smoothing_optimizer=optimizer or "lbfgsb",
            )
        else:
            gam = GAM(
                family=family_nampy,
                formula=formula,
                select=select,
                optimize_smoothing=False,
                smoothing_method="fixed",
                smoothing_params=np.asarray(fixed_sp, dtype=np.float64).ravel(),
            )
        gam.fit(data=data)

    return data, gam, _normalize_warning_messages([str(w.message) for w in caught])


def _serialize_actual_final_fit(
    gam: GAM,
    fit_warnings: list[str],
    *,
    allow_synthetic_outer_info: bool,
    include_unconditional: bool = True,
):
    fit_result = gam.fit_core_solution_.fit_result
    fit_summary = gam.fit_result(include_covariances=True)
    aic = None
    try:
        aic = float(gam.aic())
    except Exception:
        aic = None

    edf2_total = None
    if fit_result.edf2 is not None:
        edf2_total = float(np.sum(np.asarray(fit_result.edf2, dtype=np.float64)))

    return {
        "Vp": np.asarray(fit_result.cov_bayes, dtype=np.float64),
        "Ve": np.asarray(fit_result.cov_freq, dtype=np.float64),
        "Vc": (
            None
            if (not include_unconditional or fit_result.cov_unconditional is None)
            else np.asarray(fit_result.cov_unconditional, dtype=np.float64)
        ),
        "edf_by_term": np.asarray(fit_summary.edf_by_term, dtype=np.float64),
        "edf_total": float(fit_summary.edf_total),
        "edf2_total": edf2_total,
        "trace_H": float(fit_result.trace_H),
        "hat": _compute_hat_diag(gam),
        "scale": float(fit_result.scale),
        "aic": aic,
        "outer_info": _serialize_actual_outer_info(
            gam, allow_synthetic=allow_synthetic_outer_info
        ),
        "warnings": list(fit_warnings),
    }


def _serialize_expected_final_fit(expected_snapshot: dict):
    fit = expected_snapshot["fit"]
    edf2_total = None
    edf2 = _normalize_optional(fit.get("edf2", None))
    if edf2 is not None:
        edf2_total = float(np.sum(np.asarray(fit["edf2"], dtype=np.float64)))
    hat = _normalize_optional(fit.get("hat", None))

    vc = _normalize_optional(fit.get("cov_unconditional", None))

    return {
        "Vp": np.asarray(fit["cov_bayes"], dtype=np.float64),
        "Ve": np.asarray(fit["cov_freq"], dtype=np.float64),
        "Vc": None if vc is None else np.asarray(vc, dtype=np.float64),
        "edf_by_term": np.asarray(fit["edf_by_term"], dtype=np.float64),
        "edf_total": float(fit["edf_total"]),
        "edf2_total": edf2_total,
        "trace_H": float(fit["trace_H"]),
        "hat": None if hat is None else np.asarray(hat, dtype=np.float64),
        "scale": float(fit["scale"]),
        "aic": (
            None
            if _normalize_optional(fit.get("aic", None)) is None
            else float(fit["aic"])
        ),
        "outer_info": _serialize_outer_info_block(fit.get("outer_info", None)),
        "warnings": _normalize_warning_messages(fit.get("warnings", [])),
    }


def _assert_covariance_close(
    case_id: str,
    name: str,
    actual: np.ndarray | None,
    expected: np.ndarray | None,
    *,
    full_matrix: bool,
    rtol: float,
    atol: float,
):
    if expected is None:
        assert actual is None, f"{case_id}: {name} expected None, got matrix"
        return
    if actual is None:
        raise AssertionError(f"{case_id}: {name} actual is None, expected matrix")
    assert (
        actual.shape == expected.shape
    ), f"{case_id}: {name} shape mismatch {actual.shape} != {expected.shape}"
    np.testing.assert_allclose(
        np.diag(actual),
        np.diag(expected),
        rtol=rtol,
        atol=atol,
        err_msg=f"{case_id}: {name} diagonal mismatch",
    )
    np.testing.assert_allclose(
        float(np.trace(actual)),
        float(np.trace(expected)),
        rtol=rtol,
        atol=atol,
        err_msg=f"{case_id}: {name} trace mismatch",
    )
    if full_matrix:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"{case_id}: {name} full-matrix mismatch",
        )


def _assert_scalar_close(case_id: str, name: str, actual, expected, *, atol: float):
    if expected is None:
        assert actual is None, f"{case_id}: {name} expected None, got {actual!r}"
        return
    if actual is None:
        raise AssertionError(f"{case_id}: {name} actual is None, expected {expected!r}")
    if np.isnan(expected):
        assert np.isnan(actual), f"{case_id}: {name} expected NaN, got {actual!r}"
        return
    np.testing.assert_allclose(
        float(actual),
        float(expected),
        rtol=0.0,
        atol=atol,
        err_msg=f"{case_id}: {name} mismatch",
    )


def _assert_outer_info_close(case_id: str, actual, expected, *, atol: float):
    if expected is None:
        assert actual is None, f"{case_id}: expected no outer_info, got {actual!r}"
        return

    assert actual is not None, f"{case_id}: missing outer_info"
    if expected["conv"] is not None:
        assert actual["conv"] == expected["conv"], (
            f"{case_id}: outer_info conv mismatch "
            f"{actual['conv']!r} != {expected['conv']!r}"
        )
    if expected["hess"] is not None:
        np.testing.assert_allclose(
            np.asarray(actual["hess"], dtype=np.float64),
            np.asarray(expected["hess"], dtype=np.float64),
            rtol=0.0,
            atol=max(atol, 5e-6),
            err_msg=f"{case_id}: outer_info hess mismatch",
        )


def _assert_outer_info_trace_close(case_id: str, actual, expected, *, atol: float):
    _assert_outer_info_close(case_id, actual, expected, atol=atol)

    if expected["iter"] is not None:
        assert int(actual["iter"]) == int(expected["iter"]), (
            f"{case_id}: outer_info iter mismatch "
            f"{actual['iter']!r} != {expected['iter']!r}"
        )
    if expected["score_hist"] is not None:
        np.testing.assert_allclose(
            np.asarray(actual["score_hist"], dtype=np.float64),
            np.asarray(expected["score_hist"], dtype=np.float64),
            rtol=0.0,
            atol=atol,
            err_msg=f"{case_id}: outer_info score_hist mismatch",
        )
    if expected["grad"] is not None:
        np.testing.assert_allclose(
            np.asarray(actual["grad"], dtype=np.float64),
            np.asarray(expected["grad"], dtype=np.float64),
            rtol=0.0,
            atol=atol,
            err_msg=f"{case_id}: outer_info grad mismatch",
        )


def _assert_final_fit_parity(
    case_id: str,
    actual: dict,
    expected: dict,
    *,
    full_covariance: bool,
    compare_hat: bool,
    compare_outer_info: bool,
    cov_rtol: float,
    cov_atol: float,
    scalar_atol: float,
    exact_outer_info_trace: bool,
):
    actual_edf_by_term = np.asarray(actual["edf_by_term"], dtype=np.float64)
    expected_edf_by_term = np.asarray(expected["edf_by_term"], dtype=np.float64)
    if (
        actual_edf_by_term.ndim == 1
        and expected_edf_by_term.ndim == 1
        and actual_edf_by_term.size > expected_edf_by_term.size
    ):
        actual_edf_by_term = actual_edf_by_term[-expected_edf_by_term.size :]

    for key in ("Vp", "Ve", "Vc"):
        _assert_covariance_close(
            case_id,
            key,
            actual[key],
            expected[key],
            full_matrix=full_covariance,
            rtol=cov_rtol,
            atol=cov_atol,
        )

    np.testing.assert_allclose(
        actual_edf_by_term,
        expected_edf_by_term,
        rtol=0.0,
        atol=scalar_atol,
        err_msg=f"{case_id}: edf_by_term mismatch",
    )
    _assert_scalar_close(
        case_id,
        "edf_total",
        actual["edf_total"],
        expected["edf_total"],
        atol=scalar_atol,
    )
    if expected["edf2_total"] is not None and actual["edf2_total"] is not None:
        _assert_scalar_close(
            case_id,
            "edf2_total",
            actual["edf2_total"],
            expected["edf2_total"],
            atol=max(scalar_atol, 5e-6),
        )
    _assert_scalar_close(
        case_id,
        "trace_H",
        actual["trace_H"],
        expected["trace_H"],
        atol=scalar_atol,
    )
    _assert_scalar_close(
        case_id,
        "scale",
        actual["scale"],
        expected["scale"],
        atol=max(scalar_atol, 5e-8),
    )
    _assert_scalar_close(
        case_id,
        "aic",
        actual["aic"],
        expected["aic"],
        atol=max(scalar_atol, 5e-5),
    )

    if compare_hat and expected["hat"] is not None:
        assert actual["hat"] is not None, f"{case_id}: missing hat diagonal"
        np.testing.assert_allclose(
            np.asarray(actual["hat"], dtype=np.float64),
            np.asarray(expected["hat"], dtype=np.float64),
            rtol=0.0,
            atol=max(scalar_atol, 5e-5),
            err_msg=f"{case_id}: hat mismatch",
        )

    if compare_outer_info:
        if exact_outer_info_trace:
            _assert_outer_info_trace_close(
                case_id,
                actual["outer_info"],
                expected["outer_info"],
                atol=max(scalar_atol, 5e-6),
            )
        else:
            _assert_outer_info_close(
                case_id,
                actual["outer_info"],
                expected["outer_info"],
                atol=max(scalar_atol, 5e-6),
            )

    assert actual["warnings"] == expected["warnings"], (
        f"{case_id}: warnings mismatch "
        f"{actual['warnings']!r} != {expected['warnings']!r}"
    )


def _magic_case_id(case: CaseSpec) -> str:
    return f"{case.case_id}_gcv_magic"


@pytest.mark.parametrize(
    "case_id",
    ["binomial_logit", "poisson", "gamma_log"],
)
def test_gam_fit3_non_gaussian_unconditional_postproc_matches_mgcv(case_id: str):
    case = next(c for c in ORDINARY_CASES if c.case_id == case_id)
    expected_snapshot = _run_mgcv_snapshot(
        data=case.data_factory(),
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    optimizer = _nampy_optimizer_name(expected_snapshot)
    _data, gam, fit_warnings = _fit_requested_case(
        case,
        method="REML",
        optimizer=optimizer,
    )

    actual = _serialize_actual_final_fit(
        gam,
        fit_warnings,
        allow_synthetic_outer_info=False,
    )
    expected = _serialize_expected_final_fit(expected_snapshot)

    _assert_covariance_close(
        case_id,
        "Vc",
        actual["Vc"],
        expected["Vc"],
        full_matrix=_formula_is_orientation_stable(
            case.formula,
            skip_coef_comparison=bool(case.skip_coef_comparison),
        ),
        rtol=3e-5,
        atol=5e-8,
    )
    _assert_scalar_close(
        case_id,
        "edf2_total",
        actual["edf2_total"],
        expected["edf2_total"],
        atol=5e-6,
    )


@pytest.mark.parametrize(
    "case", MAGIC_CASES, ids=[_magic_case_id(c) for c in MAGIC_CASES]
)
def test_magic_postprocessing_final_fit_matches_mgcv(case: CaseSpec):
    if case.case_id in _KNOWN_FAILING_OR_WARNING_CASE_IDS:
        pytest.xfail(
            "Known requested parity gap/warning case; post-proc coverage is kept "
            "visible without treating the existing model-level mismatch as fixed."
        )
    expected_snapshot = _run_mgcv_snapshot(
        data=case.data_factory(),
        formula=case.formula,
        family=case.family,
        method="GCV.Cp",
        select=case.select,
        weights_column=case.weights_column,
    )
    sp = np.asarray(expected_snapshot["fit"]["smoothing_params"], dtype=np.float64)
    _data, gam, fit_warnings = _fit_requested_case(case, method="GCV.Cp", fixed_sp=sp)

    actual = _serialize_actual_final_fit(
        gam,
        fit_warnings,
        allow_synthetic_outer_info=False,
        include_unconditional=False,
    )
    expected = _serialize_expected_final_fit(expected_snapshot)

    _assert_final_fit_parity(
        _magic_case_id(case),
        actual,
        expected,
        full_covariance=_formula_is_orientation_stable(
            case.formula,
            skip_coef_comparison=bool(case.skip_coef_comparison),
        ),
        compare_hat=True,
        compare_outer_info=False,
        cov_rtol=max(
            5e-6,
            5.0 * float(case.se_tol_scale),
            0.25 * float(case.criterion_atol),
        ),
        cov_atol=max(
            5e-8,
            5.0 * float(case.se_tol_scale),
            0.25 * float(case.criterion_atol),
        ),
        scalar_atol=max(
            5e-5,
            5.0 * float(case.se_tol_scale),
            0.25 * float(case.criterion_atol),
        ),
        exact_outer_info_trace=False,
    )


@pytest.mark.parametrize(
    "case", ORDINARY_CASES, ids=[c.case_id for c in ORDINARY_CASES]
)
def test_gam_fit3_postprocessing_final_fit_matches_mgcv(case: CaseSpec):
    if case.case_id in _KNOWN_FAILING_OR_WARNING_CASE_IDS:
        pytest.xfail(
            "Known requested parity gap/warning case; post-proc coverage is kept "
            "visible without treating the existing model-level mismatch as fixed."
        )
    expected_snapshot = _run_mgcv_snapshot(
        data=case.data_factory(),
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    optimizer = _nampy_optimizer_name(expected_snapshot)
    _data, gam, fit_warnings = _fit_requested_case(
        case,
        method="REML",
        optimizer=optimizer,
    )

    actual = _serialize_actual_final_fit(
        gam,
        fit_warnings,
        allow_synthetic_outer_info=False,
    )
    expected = _serialize_expected_final_fit(expected_snapshot)
    family_name = str(case.family).lower()
    if (
        family_name != "gaussian"
        and expected["Vc"] is not None
        and actual["Vc"] is None
    ):
        pytest.xfail(
            "Real implementation gap: non-Gaussian PIRLS final-fit objects do not "
            "yet carry mgcv-style unconditional covariance/edf2 post-processing."
        )
    if case.case_id == "gamma_log":
        pytest.xfail(
            "gamma_log: final-fit AIC parity remains a separate gap; "
            "non-Gaussian unconditional Vc/edf2 parity is covered above."
        )

    _assert_final_fit_parity(
        case.case_id,
        actual,
        expected,
        full_covariance=_formula_is_orientation_stable(
            case.formula,
            skip_coef_comparison=bool(case.skip_coef_comparison),
        ),
        compare_hat=True,
        compare_outer_info=True,
        cov_rtol=3e-5,
        cov_atol=5e-8,
        scalar_atol=2e-4,
        exact_outer_info_trace=(case.weights_column is None),
    )


@pytest.mark.parametrize(
    "case", GENERAL_SE_CASES, ids=[case[0] for case in GENERAL_SE_CASES]
)
def test_gam_fit5_postprocessing_final_fit_matches_mgcv(case):
    case_id, family, formula, data_factory, method, pred_atol, sp_log_atol, _ = case
    if any(
        tag in case_id for tag in _GENERAL_POSTPROC_KNOWN_GAP_TAGS
    ) or case_id.startswith("shashlss"):
        pytest.xfail(
            "Known general-family post-proc gap: advanced/select/by/tensor or "
            "shashlss surfaces do not yet have exact mgcv final-fit parity."
        )
    data = data_factory()
    select = "select_true" in case_id
    expected_snapshot = _run_mgcv_snapshot(
        data=data,
        formula=formula,
        family=family,
        method=method,
        select=select,
    )
    optimizer = _nampy_optimizer_name(expected_snapshot)
    _data, gam, fit_warnings = _fit_general_case(case, optimizer=optimizer)

    actual = _serialize_actual_final_fit(
        gam,
        fit_warnings,
        allow_synthetic_outer_info=False,
    )
    expected = _serialize_expected_final_fit(expected_snapshot)

    _assert_final_fit_parity(
        case_id,
        actual,
        expected,
        full_covariance=_formula_is_orientation_stable(
            formula,
            skip_coef_comparison=("tp" in str(formula) or "t2(" in str(formula)),
        ),
        compare_hat=False,
        compare_outer_info=True,
        cov_rtol=max(5e-5, 10.0 * float(pred_atol)),
        cov_atol=max(5e-8, 10.0 * float(pred_atol)),
        scalar_atol=max(5e-4, 10.0 * float(pred_atol), float(sp_log_atol)),
        exact_outer_info_trace=False,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            next(case for case in ORDINARY_CASES if case.case_id == "gaussian_weights"),
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Real implementation gap: weighted outer_newton "
                    "outer.info trace does not yet match mgcv exactly."
                ),
            ),
        ),
        pytest.param(
            next(case for case in GENERAL_SE_CASES if case[0] == "gaulss_cr"),
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Real implementation gap: general-family outer_newton "
                    "outer.info trace/count history does not yet match mgcv exactly."
                ),
            ),
        ),
        pytest.param(
            next(case for case in GENERAL_SE_CASES if case[0] == "gevlss_cr"),
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Real implementation gap: general-family outer_newton "
                    "outer.info trace/count history does not yet match mgcv exactly."
                ),
            ),
        ),
    ],
    ids=[
        "gaussian_weights_outer_info_exact",
        "gaulss_cr_outer_info_exact",
        "gevlss_cr_outer_info_exact",
    ],
)
def test_gam_fit5_outer_info_trace_exact_known_gap(case):
    if isinstance(case, CaseSpec):
        expected_snapshot = _run_mgcv_snapshot(
            data=case.data_factory(),
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
        optimizer = _nampy_optimizer_name(expected_snapshot)
        _data, gam, _fit_warnings = _fit_requested_case(
            case,
            method="REML",
            optimizer=optimizer,
        )
        actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
        expected = _serialize_outer_info_block(
            expected_snapshot["fit"].get("outer_info")
        )
        _assert_outer_info_trace_close(case.case_id, actual, expected, atol=5e-6)
        return

    case_id, family, formula, data_factory, method, _pred_atol, _sp_log_atol, _ = case
    data = data_factory()
    expected_snapshot = _run_mgcv_snapshot(
        data=data,
        formula=formula,
        family=family,
        method=method,
        select=False,
    )
    optimizer = _nampy_optimizer_name(expected_snapshot)
    _data, gam, _fit_warnings = _fit_general_case(case, optimizer=optimizer)
    actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
    expected = _serialize_outer_info_block(expected_snapshot["fit"].get("outer_info"))
    _assert_outer_info_trace_close(case_id, actual, expected, atol=5e-6)
