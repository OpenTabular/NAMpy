"""Probe Poisson REML optim lifecycle parity metadata."""

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
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)


def main() -> None:
    case = next(c for c in OPTIMIZATION_LIFECYCLE_CASES if c.case_id == "poisson_reml_optim_two_cr")
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
    print("actual fit outer_info:", actual_trace["fit"]["outer_info"])
    print("expected fit outer_info:", expected_trace["fit"]["outer_info"])
    print("actual final Vc is None:", gam.gam_result_.fit_core_solution.fit_result.cov_unconditional is None)
    print("expected final Vc is None:", expected_snapshot["fit"].get("cov_unconditional", None) is None)
    print("actual smoothing params:", np.asarray(gam.smoothing_params))
    print(
        "expected smoothing params:",
        np.asarray(expected_snapshot["fit"]["smoothing_params"]),
    )
    print("warnings:", fit_warnings)
    print("nfev/njev:", gam._optim_result.nfev, gam._optim_result.njev)
    print("trace optimizer fields:", actual_trace["fit"]["outer_info"].get("optimizer"))
    print("trace counts:", actual_trace["fit"]["outer_info"].get("counts"))
    print("expected counts:", expected_trace["fit"]["outer_info"].get("counts"))
    print("actual outer_info keys:", sorted(actual_trace["fit"]["outer_info"].keys()))
    print("expected outer_info keys:", sorted(expected_trace["fit"]["outer_info"].keys()))
    print("smoothing params:", np.asarray(gam.smoothing_params))
    print("actual trace row0:", actual_trace["trace"][0])
    print("expected trace row0:", expected_trace["trace"][0])
    print("actual trace row_last:", actual_trace["trace"][-1])
    print("expected trace row_last:", expected_trace["trace"][-1])

    def _first_mismatch(actual, expected, path="root"):
        if expected is None or actual is None:
            return None if actual == expected else (path, actual, expected)
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return path, actual, expected
            for key in expected:
                if key not in actual:
                    return f"{path}.{key}", None, expected[key]
                mismatch = _first_mismatch(actual[key], expected[key], f"{path}.{key}")
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(expected, list):
            if not isinstance(actual, list):
                return path, actual, expected
            if len(actual) != len(expected):
                return f"{path}.len", len(actual), len(expected)
            for i, (a, e) in enumerate(zip(actual, expected)):
                mismatch = _first_mismatch(a, e, f"{path}[{i}]")
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(expected, np.ndarray):
            actual_arr = np.asarray(actual)
            if actual_arr.shape != expected.shape:
                return f"{path}.shape", actual_arr.shape, expected.shape
            if not np.allclose(actual_arr, expected, rtol=0.0, atol=1e-12):
                return path, actual_arr, expected
            return None
        if actual != expected:
            return path, actual, expected
        return None

    mismatch = _first_mismatch(actual_trace["fit"]["outer_info"], expected_trace["fit"]["outer_info"])
    print("first outer-info mismatch:", mismatch)

    trace_mismatch = _first_mismatch(actual_trace["trace"], expected_trace["trace"])
    print("first trace mismatch:", trace_mismatch)


if __name__ == "__main__":
    main()
