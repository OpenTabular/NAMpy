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
    _assert_serialized_trace_matches_mgcv(
        actual_trace,
        expected_trace,
        atol=case.trace_atol,
        sp_atol=case.trace_sp_atol or case.trace_atol,
    )

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
    _assert_final_fit_parity(
        case.case_id,
        actual_final,
        expected_final,
        full_covariance=final_fit_uses_exact_orientation_parity(
            case.formula,
            skip_coef_comparison=case.skip_coef_comparison,
        ),
        compare_hat=case.compare_hat,
        compare_outer_info=True,
        cov_rtol=case.cov_rtol,
        cov_atol=case.cov_atol,
        scalar_atol=case.scalar_atol,
        exact_outer_info_trace=case.exact_outer_info_trace,
    )
