"""Probe the Gamma joint-scale BFGS lifecycle trace mismatch."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam import GAM
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.smoothing_selection.optimize.bfgs_strict import (
    _eval_objective_at,
    _finite_difference_initial_inverse_hessian,
)
from nampy.gam.smoothing_selection.optimize.objectives import _JointGammaPirlsRemlObjective
from tests._paths import REPO_ROOT
from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace


def _case():
    for item in OPTIMIZATION_LIFECYCLE_CASES:
        if item.case_id == "gamma_reml_bfgs_cr_known_gap":
            return item
    raise RuntimeError("gamma_reml_bfgs_cr_known_gap not found")


def _arr(value):
    if value is None or value == {}:
        return None
    return np.asarray(value, dtype=np.float64)


def main() -> None:
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
                    str(REPO_ROOT / "debug" / "gamma_bfgs_initial_probe.R"),
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
    expected = _run_mgcv_outer_trace(
        data,
        str(case.formula),
        case.mgcv_family,
        case.method,
        case.optimizer,
        select=case.select,
    )
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer=case.smoothing_optimizer or "outer_newton",
        **dict(case.gam_kwargs),
    )
    gam.fit(data=data)
    actual = build_optimizer_trace(gam)

    gam0 = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=False,
        smoothing_method=case.method,
        **dict(case.gam_kwargs),
    )
    gam0.fit(data=data)
    y0 = gam0.family.validate_y(gam0.y_)
    init_sp = _initial_smoothing_params_from_design(gam0, y0)
    init_log_sp = np.log(np.asarray(init_sp, dtype=np.float64).ravel())
    mu_null = np.repeat(float(np.mean(np.asarray(y0, dtype=np.float64).ravel())), gam0.n_samples_)
    null_scale = float(gam0.family.deviance(np.asarray(y0, dtype=np.float64).ravel(), mu_null)) / float(gam0.n_samples_)
    phi0 = max(null_scale / 10.0, 1e-12)
    x0 = np.concatenate([init_log_sp, np.array([np.log(phi0)], dtype=np.float64)])
    obj0 = _JointGammaPirlsRemlObjective(gam0, y0, "REML")
    score0, grad0, _, _, coef0, eta0, mu0, _ = _eval_objective_at(
        obj0,
        x0,
        need_grad=True,
        commit_start=True,
    )
    B0 = _finite_difference_initial_inverse_hessian(
        obj0,
        x0,
        np.asarray(grad0, dtype=np.float64),
        start_coef=coef0,
        start_eta=eta0,
        start_mu=mu0,
    )
    print("actual_initial_x0", x0)
    print("actual_initial_score", score0)
    print("actual_initial_grad", grad0)
    print("actual_initial_B", B0)

    print("expected_fit", expected["fit"])
    print("actual_fit", actual["fit"])
    print("expected_trace_len", len(expected["trace"]))
    print("actual_trace_len", len(actual["trace"]))
    for i, (a_row, e_row) in enumerate(zip(actual["trace"], expected["trace"], strict=True)):
        print("row", i)
        print("actual", a_row)
        print("expected", e_row)
        for key in ("log_sp", "log_scale", "criterion", "gradient", "gradient_full"):
            a_val = _arr(a_row.get(key))
            e_val = _arr(e_row.get(key))
            if a_val is None or e_val is None:
                continue
            print(key, "diff", a_val - e_val)


if __name__ == "__main__":
    main()
