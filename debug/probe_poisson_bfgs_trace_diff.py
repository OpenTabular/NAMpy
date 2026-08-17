"""Probe exact Poisson REML BFGS trace parity failure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ruff: noqa: E402, I001



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nampy.gam.model.api import GAM
from nampy.gam.parity import build_optimizer_trace
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _make_poisson_data,
    _run_mgcv_outer_trace,
)


def _arr(value):
    return np.asarray(value, dtype=np.float64).ravel()


def main() -> None:
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "bfgs")

    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
    )
    gam.fit(data=data)

    raw_trace = list(getattr(gam, "_optim_trace", []) or [])
    serialized = build_optimizer_trace(gam)
    expected_trace = list(expected["trace"])

    print("raw rows", len(raw_trace), "expected rows", len(expected_trace))
    for i, (actual, exp) in enumerate(zip(raw_trace, expected_trace)):
        a = _arr(actual["log_sp"])
        e = _arr(exp["log_sp"])
        print("row", i, "log_sp diff", a - e, "max", np.max(np.abs(a - e)))
        print(
            "row",
            i,
            "criterion diff",
            float(actual["criterion"]) - float(exp["criterion"]),
        )

    actual_fit_log_sp = np.log(_arr(serialized["fit"]["smoothing_params"]))
    expected_fit_log_sp = np.log(_arr(expected["fit"]["smoothing_params"]))
    print("fit log_sp actual", actual_fit_log_sp)
    print("fit log_sp expect", expected_fit_log_sp)
    print("fit log_sp diff", actual_fit_log_sp - expected_fit_log_sp)
    print("serialized last", serialized["trace"][-1])
    print("expected last", expected_trace[-1])
    print("outer actual", serialized["fit"]["outer_info"])
    print("outer expect", expected["fit"]["outer_info"])


if __name__ == "__main__":
    main()
