from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gam_cartesian_matrix import make_data

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)
from tests.smooths.test_mgcv_smoothcon_parity import _compile_formula_design

FORMULA = (
    'y ~ s(f, x0, bs="sz", k=7, m=2, xt=list(bs="ps"), '
    "sp=c(1.0,1.2,1.4,1.6))"
)
SMOOTH = (
    's(f, x0, bs="sz", k=7, m=2, xt=list(bs="ps"), '
    "sp=c(1.0,1.2,1.4,1.6))"
)
FAMILY = {"name": "gaussian", "link": "inverse"}


def main() -> None:
    data = make_data("positive")
    design = _compile_formula_design(data, FORMULA)
    expected_X = np.asarray(_run_mgcv_smoothcon_matrix(data, SMOOTH)["X"], dtype=float)
    actual_X = np.asarray(design.design_matrix, dtype=float)
    print("X shape", actual_X.shape, expected_X.shape)
    print("X max abs", np.max(np.abs(actual_X - expected_X)))

    expected_S = _run_mgcv_smoothcon_penalties(
        data,
        SMOOTH,
        absorb_cons=True,
        scale_penalty=True,
    )["S"]
    actual_S = [np.asarray(pb.matrix, dtype=float) for pb in design.compiled_penalties]
    print("S count", len(actual_S), len(expected_S))
    for i, (got, want) in enumerate(zip(actual_S, expected_S)):
        want = np.asarray(want, dtype=float)
        print(i, got.shape, want.shape, np.max(np.abs(got - want)))

    gam = GAM(
        family=FAMILY,
        formula=FORMULA,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        FORMULA,
        FAMILY,
        "fixed",
        allow_live_run=True,
    )
    print("loglik", gam.loglik(), expected["fit"]["loglik"])
    print("deviance", actual["fit"]["deviance"], expected["fit"]["deviance"])
    print("penalty", actual["fit"].get("penalty_quadratic"), expected["fit"].get("penalty_quadratic"))
    print("edf", actual["fit"]["edf_total"], expected["fit"]["edf_total"])
    fit_result = gam.gam_result_.fit_core_solution.fit_result
    print("inner trace", fit_result.inner_trace)
    print("iter", fit_result.iter, fit_result.converged, fit_result.failure_reason)
    print("coef max abs", np.max(np.abs(np.asarray(actual["fit"]["coef_full"], dtype=float) - np.asarray(expected["fit"]["coef_full"], dtype=float))))
    print(
        "response max abs",
        np.max(
            np.abs(
                np.asarray(actual["predictions"]["response"], dtype=float)
                - np.asarray(expected["predictions"]["response"], dtype=float)
            )
        ),
    )
    print("sp", gam.smoothing_params, expected["fit"].get("smoothing_params"))


if __name__ == "__main__":
    main()
