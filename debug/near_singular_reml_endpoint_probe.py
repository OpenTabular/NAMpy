"""Trace the noiseless intercept + random-effect REML boundary endpoint.

This is a preserved diagnostic for the case covered by
``test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv``.  Upstream
``mgcv::newton`` stops with a step failure at iteration 51 for this data; the
probe caps NAMpy just beyond that point and prints the accepted outer trace.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.fit.selection.optimize import driver as driver_module
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)


def _data() -> pd.DataFrame:
    f = np.array(["b", "a", "c", "a", "b", "c", "a", "c"], dtype=object)
    effects = {"a": 1.5, "b": -0.25, "c": 0.75}
    y = np.array([effects[value] for value in f], dtype=np.float64)
    return pd.DataFrame({"y": y, "f": f})


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _focused_trace(rows):
    fields = (
        "iter",
        "log_sp",
        "log_scale",
        "criterion",
        "gradient_full",
        "hessian_full",
        "accepted_step_norm",
        "rank_info",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in rows
        if 12 <= int(row.get("iter", 0)) <= 25
    ]


def main() -> None:
    data = _data()
    formula = 'y ~ s(f, bs="re")'
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    expected_trace = _run_mgcv_outer_trace(
        data, formula, "gaussian", "REML", "newton"
    )

    original_newton = driver_module.optimize_outer_newton_indefinite_hessian

    def capped_newton(*args, **kwargs):
        kwargs["max_iter"] = 60
        return original_newton(*args, **kwargs)

    driver_module.optimize_outer_newton_indefinite_hessian = capped_newton
    try:
        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=True,
            smoothing_method="REML",
            smoothing_optimizer="outer_newton",
        )
        gam.fit(data=data)
    finally:
        driver_module.optimize_outer_newton_indefinite_hessian = original_newton

    result = gam._optim_result
    payload = {
        "mgcv": {
            "log_sp": expected["fit"]["log_smoothing_params"],
            "edf_by_term": expected["fit"]["edf_by_term"],
            "outer_info": expected["fit"]["outer_info"],
            "trace": _focused_trace(expected_trace["trace"]),
        },
        "nampy": {
            "message": str(result.message),
            "nit": int(result.nit),
            "log_sp": np.log(np.asarray(gam.smoothing_params, dtype=np.float64)),
            "edf_by_term": np.asarray(gam.fit_result().edf_by_term, dtype=np.float64),
            "trace": _focused_trace(gam._optim_trace),
        },
    }
    print(json.dumps(_serializable(payload), indent=2))


if __name__ == "__main__":
    main()
