"""Probe estimated-theta negative-binomial BFGS initialization parity."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam import GAM
from nampy.gam.fit.selection.optimize import bfgs_strict as bfgs_module
from nampy.gam.parity import build_optimizer_trace
from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES
from tests._paths import REPO_ROOT
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)


def _case():
    return next(
        case
        for case in OPTIMIZATION_LIFECYCLE_CASES
        if case.case_id == "negbin_est_reml_bfgs_joint_theta_cr"
    )


def main() -> None:
    np.set_printoptions(precision=17)
    case = _case()
    data = case.data_factory()
    rscript = shutil.which("Rscript")
    if rscript is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            data.to_csv(csv_path, index=False)
            proc = subprocess.run(
                [
                    rscript,
                    str(REPO_ROOT / "debug" / "negbin_bfgs_initial_probe.R"),
                    str(csv_path),
                    str(case.formula),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            print("mgcv_initial_probe_stdout")
            print(proc.stdout)
            print("mgcv_initial_probe_stderr")
            print(proc.stderr)
            proc.check_returncode()

    captured = {}
    original = bfgs_module._finite_difference_initial_inverse_hessian

    def _capture(objective, x0, grad0, **kwargs):
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["grad0"] = np.asarray(grad0, dtype=np.float64).copy()
        inverse_hessian = original(objective, x0, grad0, **kwargs)
        captured["inverse_hessian"] = np.asarray(
            inverse_hessian, dtype=np.float64
        ).copy()
        captured["hessian"] = np.linalg.inv(captured["inverse_hessian"])
        return inverse_hessian

    bfgs_module._finite_difference_initial_inverse_hessian = _capture
    try:
        gam = GAM(
            family=case.family,
            formula=case.formula,
            optimize_smoothing=True,
            smoothing_method=case.method,
            smoothing_optimizer="bfgs",
        )
        gam.fit(data=data)
    finally:
        bfgs_module._finite_difference_initial_inverse_hessian = original

    expected = _run_mgcv_outer_trace(
        data,
        str(case.formula),
        case.mgcv_family,
        case.method,
        case.optimizer,
    )
    actual = build_optimizer_trace(gam)
    print("nampy_initial", captured)
    print("mgcv_first_trace", expected["trace"][0])
    print("nampy_first_trace", actual["trace"][0])
    print("mgcv_fit", expected["fit"])
    print("nampy_fit", actual["fit"])


if __name__ == "__main__":
    main()
