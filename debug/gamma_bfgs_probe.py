"""Probe Gamma joint-scale BFGS ``gdi1`` initialization parity."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam.model.api import GAM
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.smoothing_selection.criteria.pirls import (
    derivatives as derivatives_module,
)
from nampy.gam.smoothing_selection.optimize import bfgs_strict as bfgs_strict_module
from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.smoothing_selection.optimize.bfgs_strict import (
    _eval_objective_at,
    _finite_difference_initial_inverse_hessian,
)
from nampy.gam.smoothing_selection.optimize.objectives import (
    _JointGammaPirlsRemlObjective,
)
from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES
from tests._paths import REPO_ROOT
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace


def _case():
    for item in OPTIMIZATION_LIFECYCLE_CASES:
        if item.case_id == "gamma_reml_bfgs_joint_scale_cr":
            return item
    raise RuntimeError("gamma_reml_bfgs_joint_scale_cr not found")


def _arr(value):
    if value is None or value == {}:
        return None
    return np.asarray(value, dtype=np.float64)


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
    actual_initial_capture = {}
    original_initial_hessian = bfgs_strict_module._finite_difference_initial_inverse_hessian
    original_gdi1_kernel = derivatives_module._gdi1_kernel

    def _capture_gdi1_kernel(*args, **kwargs):
        result = original_gdi1_kernel(*args, **kwargs)
        if "kernel" not in actual_initial_capture:
            sp = np.asarray(args[3], dtype=np.float64)
            beta = np.asarray(result.current.beta, dtype=np.float64)
            dbeta = np.column_stack(result.ift.dbeta)
            E = derivatives_module._drop_permute_columns(
                np.asarray(result.current.canonical.Sr, dtype=np.float64),
                result.current.dropped_column_indices,
                result.current.pivot1,
            )
            root = result.ift.root_blocks[0]
            root_work = root.T @ beta * sp[0]
            Skb = root @ root_work
            Sb = E.T @ (E @ beta)
            actual_initial_capture["kernel"] = {
                "D1": np.asarray(result.D1, dtype=np.float64).copy(),
                "bSb1": np.asarray(result.bSb1, dtype=np.float64).copy(),
                "Dp1": np.asarray(result.D1 + result.bSb1, dtype=np.float64).copy(),
                "dbeta": np.column_stack(result.ift.dbeta),
                "direct_bSb1": float(beta @ Skb),
                "indirect_bSb1": float(2.0 * dbeta[:, 0] @ Sb),
                "beta": np.asarray(result.current.beta, dtype=np.float64).copy(),
                "E": np.asarray(result.current.canonical.Sr, dtype=np.float64).copy(),
                "rS": [
                    np.asarray(root, dtype=np.float64).copy()
                    for root in result.ift.root_blocks
                ],
            }
        return result

    def _capture_initial_hessian(objective, x0, grad0, **kwargs):
        gamma_state = getattr(objective.model, "_pirls_reml_gamma_state_", None)
        actual_initial_capture["x0"] = np.asarray(x0, dtype=np.float64).copy()
        actual_initial_capture["grad0"] = np.asarray(grad0, dtype=np.float64).copy()
        actual_initial_capture["inner_trace"] = list(
            getattr(objective.model, "_pirls_last_inner_trace_", []) or []
        )
        if isinstance(gamma_state, dict):
            actual_initial_capture["Dp1"] = np.asarray(
                gamma_state.get("Dp1"), dtype=np.float64
            ).copy()
            actual_initial_capture["K1"] = np.asarray(
                gamma_state.get("K1"), dtype=np.float64
            ).copy()
        result = original_initial_hessian(objective, x0, grad0, **kwargs)
        actual_initial_capture["B"] = np.asarray(result, dtype=np.float64).copy()
        start_coef = kwargs.get("start_coef")
        actual_initial_capture["start_coef"] = (
            None
            if start_coef is None
            else np.asarray(start_coef, dtype=np.float64).copy()
        )
        return result

    bfgs_strict_module._finite_difference_initial_inverse_hessian = _capture_initial_hessian
    derivatives_module._gdi1_kernel = _capture_gdi1_kernel
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer=case.smoothing_optimizer or "outer_newton",
        **dict(case.gam_kwargs),
    )
    try:
        gam.fit(data=data)
    finally:
        bfgs_strict_module._finite_difference_initial_inverse_hessian = original_initial_hessian
        derivatives_module._gdi1_kernel = original_gdi1_kernel
    actual = build_optimizer_trace(gam)
    print("actual_optimizer_initial_capture", actual_initial_capture)

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
    raw_hessian0 = np.zeros((x0.size, x0.size), dtype=np.float64)
    fdgrad0 = np.zeros(x0.size, dtype=np.float64)
    for i in range(x0.size):
        x1 = x0.copy()
        x1[i] += 1e-4
        score1, grad1, _, _, _, _, _, _ = _eval_objective_at(
            obj0,
            x1,
            start_coef=coef0,
            start_eta=eta0,
            start_mu=mu0,
            need_grad=True,
            commit_start=False,
        )
        raw_hessian0[i, :] = (np.asarray(grad1) - np.asarray(grad0)) / 1e-4
        fdgrad0[i] = (float(score1) - float(score0)) / 1e-4
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
    print("actual_initial_inner_trace", gam0._pirls_last_inner_trace_)
    print("actual_initial_raw_hessian", raw_hessian0)
    print("actual_initial_fdgrad", fdgrad0)
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
