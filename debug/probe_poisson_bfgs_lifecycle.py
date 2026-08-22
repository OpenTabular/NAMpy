"""Probe Poisson REML BFGS lifecycle parity metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nampy.gam.parity import build_optimizer_trace
from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_optimization_lifecycle_parity import (
    _fit_lifecycle_case,
)
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace


def main() -> None:
    case = next(
        c for c in OPTIMIZATION_LIFECYCLE_CASES if c.case_id == "poisson_reml_bfgs_two_cr"
    )
    data, gam, fit_warnings = _fit_lifecycle_case(case)
    expected_trace = _run_mgcv_outer_trace(
        data,
        str(case.formula),
        case.mgcv_family,
        case.method,
        case.optimizer,
        select=case.select,
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
    actual_trace = build_optimizer_trace(gam)
    print("actual trace rows:", len(actual_trace["trace"]))
    print("expected trace rows:", len(expected_trace["trace"]))
    print("actual final smoothing params:", np.asarray(gam.smoothing_params))
    print("expected final smoothing params:", np.asarray(expected_snapshot["fit"]["smoothing_params"]))
    print("actual optim x:", np.asarray(gam._optim_result.x))
    print("actual raw trace len:", len(getattr(gam._optim_result, "optim_trace", []) or []))
    if getattr(gam._optim_result, "optim_trace", None):
        print("actual raw last trace row:", gam._optim_result.optim_trace[-1])
    print("actual final Vc is None:", gam.gam_result_.fit_core_solution.fit_result.cov_unconditional is None)
    print("expected final Vc is None:", expected_snapshot["fit"].get("cov_unconditional", None) is None)
    print("actual outer_info:", actual_trace["fit"]["outer_info"])
    print("expected outer_info:", expected_trace["fit"]["outer_info"])
    print("warnings:", fit_warnings)
    print("last actual trace row:", actual_trace["trace"][-1])
    print("last expected trace row:", expected_trace["trace"][-1])


if __name__ == "__main__":
    main()
