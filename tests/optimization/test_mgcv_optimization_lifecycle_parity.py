import warnings

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.parity import build_optimizer_trace
from tests._optimization_lifecycle_registry import (
    OPTIMIZATION_LIFECYCLE_CASES,
    OptimizationLifecycleCase,
)
from tests.mgcv_invariant_policy import final_fit_uses_exact_orientation_parity
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _assert_serialized_trace_matches_mgcv,
    _run_mgcv_outer_trace,
)
from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    _assert_final_fit_parity,
    _normalize_warning_messages,
    _serialize_actual_final_fit,
    _serialize_expected_final_fit,
)

pytestmark = [pytest.mark.surface_trace]


def _exchangeable_sp_permutation(rows, case: OptimizationLifecycleCase):
    """
    Canonical sp permutation for mgcv-indeterminate exchangeable groups.

    Upstream assigns one sp per nat.param null column, but R's eigen orders
    those numerically-zero eigenvalues by roundoff — mgcv itself flips the
    order under a row permutation of the same data
    (the retained local null-order stability probe). Each side is therefore
    canonicalized independently by descending final log-sp inside each
    declared group; every number must still match strictly afterwards.
    """
    if not case.exchangeable_sp_groups or not rows:
        return None
    last_lsp = np.asarray(rows[-1]["log_sp"], dtype=np.float64).ravel()
    perm = np.arange(last_lsp.size, dtype=int)
    for group in case.exchangeable_sp_groups:
        idx = np.asarray(group, dtype=int)
        perm[idx] = idx[np.argsort(-last_lsp[idx], kind="stable")]
    if np.array_equal(perm, np.arange(last_lsp.size)):
        return None
    return perm


def _exchangeable_coef_permutation(perm, case: OptimizationLifecycleCase, n_coef):
    if perm is None or not case.exchangeable_sp_coef_cols:
        return None
    coef_perm = np.arange(int(n_coef), dtype=int)
    for group, col_sets in zip(
        case.exchangeable_sp_groups, case.exchangeable_sp_coef_cols, strict=True
    ):
        group = list(group)
        for pos_in_group, sp_target in enumerate(group):
            sp_source = int(perm[sp_target])
            target_cols = np.asarray(col_sets[pos_in_group], dtype=int)
            source_cols = np.asarray(col_sets[group.index(sp_source)], dtype=int)
            coef_perm[target_cols] = source_cols
    if np.array_equal(coef_perm, np.arange(int(n_coef))):
        return None
    return coef_perm


def _extend_perm(perm, size):
    """Extend an sp-block permutation with identity for trailing joint
    coordinates (log scale / log theta are appended after the sp block)."""
    size = int(size)
    if size < perm.size:
        return None
    if size == perm.size:
        return perm
    return np.concatenate([perm, np.arange(perm.size, size, dtype=int)])


def _permute_sp_vector(value, perm):
    if value is None:
        return value
    arr = np.asarray(value, dtype=np.float64).ravel()
    full = _extend_perm(perm, arr.size)
    if full is None:
        return value
    return arr[full].tolist()


def _permute_sp_matrix(value, perm):
    if value is None:
        return value
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return value
    full = _extend_perm(perm, arr.shape[0])
    if full is None:
        return value
    return arr[np.ix_(full, full)].tolist()


def _canonicalize_outer_info_sp(outer, perm):
    if not outer:
        return
    for key in ("grad", "gradient", "gradient_full"):
        if outer.get(key) is not None:
            outer[key] = _permute_sp_vector(outer[key], perm)
    for key in ("hess", "hessian", "hessian_full"):
        if outer.get(key) is not None:
            permuted = _permute_sp_matrix(
                np.asarray(outer[key], dtype=np.float64), perm
            )
            outer[key] = np.asarray(permuted, dtype=np.float64)


def _canonicalize_exchangeable_sp(trace_payload, final_fit, case):
    """Apply the canonical exchangeable-group order to one side, in place."""
    trace_rows = list(trace_payload.get("trace", []) or [])
    perm = _exchangeable_sp_permutation(trace_rows, case)
    if perm is None:
        return
    for row in trace_rows:
        for key in ("log_sp", "gradient", "gradient_full"):
            if key in row:
                row[key] = _permute_sp_vector(row[key], perm)
        for key in ("hessian", "hessian_full"):
            if key in row:
                row[key] = _permute_sp_matrix(row[key], perm)
    fit_block = trace_payload.get("fit")
    if isinstance(fit_block, dict):
        for key in ("smoothing_params", "log_smoothing_params"):
            if fit_block.get(key) is not None:
                fit_block[key] = _permute_sp_vector(fit_block[key], perm)
        outer = fit_block.get("outer_info")
        if isinstance(outer, dict):
            _canonicalize_outer_info_sp(outer, perm)
    if final_fit is not None:
        outer = final_fit.get("outer_info")
        if isinstance(outer, dict):
            _canonicalize_outer_info_sp(outer, perm)
        for key in ("Vp", "Ve", "Vc"):
            mat = final_fit.get(key)
            if mat is None:
                continue
            mat = np.asarray(mat, dtype=np.float64)
            coef_perm = _exchangeable_coef_permutation(perm, case, mat.shape[0])
            if coef_perm is not None:
                final_fit[key] = mat[np.ix_(coef_perm, coef_perm)]


def _lifecycle_case_param(case: OptimizationLifecycleCase):
    marks = []
    if case.status == "known_gap":
        marks.extend(
            [
                pytest.mark.status_known_gap,
                pytest.mark.xfail(strict=True, reason=case.known_gap_reason),
            ]
        )
    return pytest.param(case, id=case.case_id, marks=marks)


def _fit_lifecycle_case(case: OptimizationLifecycleCase):
    data = case.data_factory()
    sample_weight = None
    if case.weights_column is not None:
        sample_weight = np.asarray(data[case.weights_column], dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = GAM(
            family=case.family,
            formula=case.formula,
            select=case.select,
            optimize_smoothing=True,
            smoothing_method=case.method,
            smoothing_optimizer=case.smoothing_optimizer or "outer_newton",
            **dict(case.gam_kwargs),
        )
        gam.fit(data=data, sample_weight=sample_weight)

    return data, gam, _normalize_warning_messages([str(w.message) for w in caught])


@pytest.mark.parametrize(
    "case",
    [_lifecycle_case_param(case) for case in OPTIMIZATION_LIFECYCLE_CASES],
)
def test_supported_optimization_lifecycle_matches_mgcv(case: OptimizationLifecycleCase):
    """Verify that each supported optimizer branch matches mgcv across trace and final fit."""
    data, gam, fit_warnings = _fit_lifecycle_case(case)

    expected_trace = _run_mgcv_outer_trace(
        data,
        str(case.formula),
        case.mgcv_family,
        case.method,
        case.optimizer,
        select=case.select,
        weights_column=case.weights_column,
    )
    actual_trace = build_optimizer_trace(gam)

    expected_snapshot = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method=case.method,
        select=case.select,
        weights_column=case.weights_column,
        optimizer=case.optimizer,
    )
    actual_final = _serialize_actual_final_fit(
        gam,
        fit_warnings,
        allow_synthetic_outer_info=False,
    )
    expected_final = _serialize_expected_final_fit(expected_snapshot)

    _canonicalize_exchangeable_sp(actual_trace, actual_final, case)
    _canonicalize_exchangeable_sp(expected_trace, expected_final, case)
    if not case.compare_unconditional:
        # edf2 is rowSums(Vc * crossprod(R)) — a functional of Vc — so it
        # inherits the same indeterminacy branch (mgcv row-permuted edf2_total
        # 15.9940558 equals NAMpy to 7 digits; branch spread 2.4e-3).
        actual_final["Vc"] = None
        expected_final["Vc"] = None
        actual_final["edf2_total"] = None
        expected_final["edf2_total"] = None
        # AIC's effective df is sum(edf2) (logLik.gam), so the observed AIC
        # difference is exactly 2x the edf2 branch spread.
        actual_final["aic"] = None
        expected_final["aic"] = None

    _assert_serialized_trace_matches_mgcv(
        actual_trace,
        expected_trace,
        atol=case.trace_atol,
        sp_atol=case.trace_sp_atol or case.trace_atol,
    )
    _assert_final_fit_parity(
        case.case_id,
        actual_final,
        expected_final,
        full_covariance=final_fit_uses_exact_orientation_parity(
            gam,
            skip_coef_comparison=case.skip_coef_comparison,
        ),
        compare_hat=case.compare_hat,
        compare_outer_info=True,
        cov_rtol=case.cov_rtol,
        cov_atol=case.cov_atol,
        scalar_atol=case.scalar_atol,
        exact_outer_info_trace=case.exact_outer_info_trace,
    )
