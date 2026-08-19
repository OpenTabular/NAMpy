"""Probe internal BFGS evaluation state for Poisson REML lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nampy.gam.fit.selection.optimize.bfgs_mgcv as bfgs_mod
from nampy.gam.fit.selection.optimize.bfgs_mgcv import _optimize_outer_bfgs_mgcv

from nampy.gam.fit.selection.optimize.basics import (
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.fit.selection.optimize.objectives import _CriterionObjective
from nampy.gam.model_state import _fit_workspace
from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES
from tests.optimization.test_mgcv_optimization_lifecycle_parity import (
    _fit_lifecycle_case,
)


def main() -> None:
    case = next(
        c for c in OPTIMIZATION_LIFECYCLE_CASES if c.case_id == "poisson_reml_bfgs_two_cr"
    )
    data, gam, _ = _fit_lifecycle_case(case)
    y = gam.family.validate_y(gam.y_)
    init = _initial_smoothing_params_mgcv_style(gam, y)
    assert init is not None
    x0 = np.log(np.maximum(np.asarray(init, dtype=np.float64), 1e-300))
    bounds = [(-80.0, 25.0) for _ in range(int(np.asarray(x0).size))]
    objective = _CriterionObjective(gam, y, method="reml", use_gradient=True)
    calls = []
    original_eval = bfgs_mod._eval_objective_at

    def wrapped_eval(*args, **kwargs):
        x_eval = np.asarray(args[1], dtype=np.float64).ravel()
        out = original_eval(*args, **kwargs)
        score, grad, hess, dvkk, coef, eta, mu, scale = out
        kernel_state = _fit_workspace(objective.model).get("pirls_reml_derivative_kernel_state")
        raw_dvkk = None
        raw_dvkk_shape = None
        if isinstance(kernel_state, dict) and kernel_state.get("dVkk", None) is not None:
            raw_dvkk = np.asarray(kernel_state["dVkk"], dtype=np.float64)
            raw_dvkk_shape = raw_dvkk.shape
        calls.append(
            {
                "x": x_eval.copy(),
                "score": score,
                "grad": None if grad is None else np.asarray(grad, dtype=np.float64).copy(),
                "dvkk": np.asarray(dvkk, dtype=np.float64).copy(),
                "raw_dvkk_shape": raw_dvkk_shape,
                "raw_dvkk": None if raw_dvkk is None else raw_dvkk.copy(),
                "need_grad": bool(kwargs.get("need_grad", False)),
                "commit_start": bool(kwargs.get("commit_start", False)),
            }
        )
        return out

    bfgs_mod._eval_objective_at = wrapped_eval
    try:
        res = _optimize_outer_bfgs_mgcv(
            objective,
            x0,
            bounds=bounds,
            score_type="reml",
        )
    finally:
        bfgs_mod._eval_objective_at = original_eval

    print("result x:", np.asarray(res.x))
    print("result nit:", res.nit)
    print("calls:", len(calls))
    for row in calls[-12:]:
        print(row)


if __name__ == "__main__":
    main()
