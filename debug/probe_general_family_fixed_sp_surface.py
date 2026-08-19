from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.fit.selection.criteria import criterion_value
from tests.families.test_general_family_mgcv_parity import GENERAL_SE_CASES
from tests.mgcv_parity_utils import (
    _family_specs,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_fixed_sp_score,
    _run_mgcv_snapshot,
)
from tests.optimization.test_mgcv_fixed_inner_fit_parity import _run_mgcv_fit5_fixed_sp
from tests._paths import PARITY_DIR
from nampy.gam.fit.solvers.general_family_solver import run_general_family_fixed_smoothing
from nampy.gam.fit.solvers.general_family_solver import build_general_family_setup_state
from nampy.gam.fit.solvers.general_newton_solver import _PenaltyRoot, _sl_ldetS, _sl_repara

R_SCRIPT = shutil.which("Rscript")
MGCV_GENERAL_PREOPT_SCRIPT = PARITY_DIR / "mgcv_general_family_preoptimization.R"


def _case_table():
    return {case[0]: case for case in GENERAL_SE_CASES}


def _copy_attr(obj, name):
    value = getattr(obj, name, None)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


@contextmanager
def _temporary_model_state(model, *, clear_starts=False, irls_tol=None, max_irls_iter=None):
    names = (
        "_pirls_eval_start_",
        "_pirls_coef_start_",
        "_pirls_last_coef_",
        "_pirls_last_eta_",
        "_pirls_last_mu_",
        "_general_family_outer_eval_cache",
    )
    prev = {name: _copy_attr(model, name) for name in names}
    prev_tol = getattr(model, "irls_tol", None)
    prev_maxit = getattr(model, "max_irls_iter", None)
    try:
        if clear_starts:
            for name in names:
                setattr(model, name, None)
        if irls_tol is not None:
            model.irls_tol = float(irls_tol)
        if max_irls_iter is not None:
            model.max_irls_iter = int(max_irls_iter)
        yield
    finally:
        for name, value in prev.items():
            setattr(model, name, value)
        if prev_tol is None:
            if hasattr(model, "irls_tol"):
                delattr(model, "irls_tol")
        else:
            model.irls_tol = prev_tol
        if prev_maxit is None:
            if hasattr(model, "max_irls_iter"):
                delattr(model, "max_irls_iter")
        else:
            model.max_irls_iter = prev_maxit


def _score_from_existing_model(model, log_sp, method):
    return float(criterion_value(model, model.y_, log_sp, method=method.lower()))


def _score_from_fresh_model(data, formula, family, log_sp, *, select, method):
    fresh = _fit_nampy_model_fixed_sp(
        data,
        formula,
        family,
        np.exp(log_sp),
        select=select,
    )
    return float(criterion_value(fresh, fresh.y_, log_sp, method=method.lower()))


def _run_mgcv_general_preopt_fixed_sp(data, formula, family, method, sp, *, select):
    _family_nampy, family_token = _family_specs(family)
    del _family_nampy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "general_preopt.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_GENERAL_PREOPT_SCRIPT),
                str(csv_path),
                str(json_path),
                str(formula),
                family_token,
                method,
                "true" if select else "false",
                json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", default="gevlss_t2_full_true", nargs="?")
    parser.add_argument("--sp-index", type=int, default=None)
    args = parser.parse_args()

    (
        _case_id,
        family,
        formula,
        data_factory,
        method,
        _pred_atol,
        _se_atol,
        _check_response_se,
    ) = _case_table()[args.case_id]
    select = "select_true" in args.case_id
    data = data_factory()
    snapshot = _run_mgcv_snapshot(data, formula, family, method, select=select)
    sp = np.asarray(snapshot["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)

    indices = range(log_sp.size) if args.sp_index is None else [int(args.sp_index)]
    report = {
        "case_id": args.case_id,
        "family": family,
        "method": method,
        "snapshot_score_center": float(snapshot["fit"]["criterion_value"]),
        "rows": [],
    }
    for idx in indices:
        step = max(1e-4, 1e-3 * (1.0 + abs(float(log_sp[idx]))))
        for mul in (-2, -1, 0, 1, 2):
            point = log_sp.copy()
            point[idx] += mul * step
            mgcv_score = float(
                _run_mgcv_fixed_sp_score(
                    data,
                    formula,
                    family,
                    method,
                    np.exp(point),
                    select=select,
                )["criterion_value"]
            )
            run = run_general_family_fixed_smoothing(
                gam,
                gam.y_,
                np.exp(point),
                weights=gam.prior_weights_,
                deriv=0,
                score_type=method,
            )
            fit = run["fit"]
            mgcv_fit5 = _run_mgcv_fit5_fixed_sp(
                data,
                formula,
                family,
                np.exp(point),
                score_type=method,
            )
            mgcv_preopt = _run_mgcv_general_preopt_fixed_sp(
                data,
                formula,
                family,
                method,
                np.exp(point),
                select=select,
            )
            actual_setup = build_general_family_setup_state(
                gam,
                np.exp(point),
                score_type=method,
            )
            rp_init = _sl_ldetS(
                actual_setup.Sl,
                rho=np.asarray(point, dtype=np.float64),
                fixed=np.zeros_like(point, dtype=bool),
                np_=actual_setup.X_initial.shape[1],
                root=True,
                Stot=True,
                deriv=0,
            )
            x_fit = _sl_repara(rp_init["rp"], np.asarray(actual_setup.X_initial, dtype=np.float64))
            E_fit = _PenaltyRoot(rp_init["E"], use_unscaled=True)
            python_start_initial = gam.family.initialize(
                gam.y_,
                x_fit,
                [np.asarray(j, dtype=int) for j in actual_setup.jj],
                offset=run["offset_list"],
                weights=np.asarray(gam.prior_weights_, dtype=np.float64),
                E=E_fit,
            )
            row = {
                "sp_index": idx,
                "mul": mul,
                "step": step,
                "mgcv_score": mgcv_score,
                "existing_score": _score_from_existing_model(gam, point, method),
                "fresh_fit_score": _score_from_fresh_model(
                    data,
                    formula,
                    family,
                    point,
                    select=select,
                    method=method,
                ),
            }
            with _temporary_model_state(gam, clear_starts=True):
                row["existing_score_clear_starts"] = _score_from_existing_model(
                    gam, point, method
                )
            with _temporary_model_state(gam, clear_starts=True):
                gam._pirls_eval_start_ = np.asarray(
                    mgcv_fit5["coefficients_full"], dtype=np.float64
                )
                row["existing_score_mgcv_start"] = _score_from_existing_model(
                    gam, point, method
                )
            with _temporary_model_state(
                gam,
                clear_starts=True,
                irls_tol=1e-10,
                max_irls_iter=max(int(getattr(gam, "max_irls_iter", 200)), 400),
            ):
                row["existing_score_clear_starts_tight_tol"] = _score_from_existing_model(
                    gam, point, method
                )
            for key in (
                "existing_score",
                "fresh_fit_score",
                "existing_score_clear_starts",
                "existing_score_mgcv_start",
                "existing_score_clear_starts_tight_tol",
            ):
                row[f"{key}_diff"] = float(row[key] - mgcv_score)
            row["fit5_score"] = float(fit["score"])
            row["fit5_iter"] = int(fit["iter"])
            row["mgcv_fit5_iter"] = int(mgcv_fit5["iter"])
            row["fit5_score_diff_vs_mgcv_fit5"] = float(
                fit["score"] - float(mgcv_fit5["score"])
            )
            row["fit5_loglik_diff"] = float(fit["l"] - float(mgcv_fit5["loglik"]))
            row["fit5_ldetHp_diff"] = float(
                fit["ldetHp"] - float(mgcv_fit5["ldetHp"])
            )
            row["fit5_coef_full_max_abs_diff"] = float(
                np.max(
                    np.abs(
                        np.asarray(fit["coef"], dtype=np.float64)
                        - np.asarray(mgcv_fit5["coefficients"], dtype=np.float64)
                    )
                )
            )
            row["setup_X_initial_max_abs_diff"] = float(
                np.max(
                    np.abs(
                        np.asarray(actual_setup.X_initial, dtype=np.float64)
                        - np.asarray(mgcv_preopt["X_initial"], dtype=np.float64)
                    )
                )
            )
            row["setup_St_max_abs_diff"] = float(
                np.max(
                    np.abs(
                        np.asarray(actual_setup.St, dtype=np.float64)
                        - np.asarray(mgcv_preopt["St"], dtype=np.float64)
                    )
                )
            )
            row["setup_ldetS_diff"] = float(
                float(actual_setup.ldetS) - float(mgcv_preopt["ldetS"])
            )
            mgcv_coef_fit = np.asarray(mgcv_fit5["coefficients"], dtype=np.float64)
            mgcv_ll_python = gam.family.ll(
                gam.y_,
                np.asarray(actual_setup.X_initial, dtype=np.float64),
                [np.asarray(j, dtype=int) for j in actual_setup.jj],
                mgcv_coef_fit,
                np.asarray(gam.prior_weights_, dtype=np.float64),
                offset=run["offset_list"],
                deriv=0,
            )["l"]
            mgcv_ll_d1 = gam.family.ll(
                gam.y_,
                np.asarray(actual_setup.X_initial, dtype=np.float64),
                [np.asarray(j, dtype=int) for j in actual_setup.jj],
                mgcv_coef_fit,
                np.asarray(gam.prior_weights_, dtype=np.float64),
                offset=run["offset_list"],
                deriv=1,
            )
            pen_grad_at_mgcv = np.asarray(mgcv_ll_d1["lb"], dtype=np.float64) - np.asarray(
                fit["St_full"], dtype=np.float64
            ) @ mgcv_coef_fit
            mgcv_lbb_python = np.asarray(mgcv_ll_d1["lbb"], dtype=np.float64)
            mgcv_lbb_expected = np.asarray(mgcv_fit5["lbb"], dtype=np.float64)
            row["python_ll_at_mgcv_coef_diff"] = float(
                float(mgcv_ll_python) - float(mgcv_fit5["loglik"])
            )
            row["python_lbb_at_mgcv_coef_max_abs_diff"] = float(
                np.max(np.abs(mgcv_lbb_python - mgcv_lbb_expected))
            )
            row["python_lbb_block_max_abs_diff_at_mgcv_coef"] = [
                [
                    float(
                        np.max(
                            np.abs(
                                mgcv_lbb_python[
                                    np.ix_(np.asarray(j_i, dtype=int), np.asarray(j_j, dtype=int))
                                ]
                                - mgcv_lbb_expected[
                                    np.ix_(np.asarray(j_i, dtype=int), np.asarray(j_j, dtype=int))
                                ]
                            )
                        )
                    )
                    for j_j in actual_setup.jj
                ]
                for j_i in actual_setup.jj
            ]
            row["python_pen_grad_max_abs_at_mgcv_coef"] = float(
                np.max(np.abs(pen_grad_at_mgcv))
            )
            row["python_pen_grad_block_max_abs_at_mgcv_coef"] = [
                float(np.max(np.abs(pen_grad_at_mgcv[np.asarray(j, dtype=int)])))
                for j in actual_setup.jj
            ]
            row["python_start_initial_max_abs_diff"] = float(
                np.max(
                    np.abs(
                        np.asarray(python_start_initial, dtype=np.float64)
                        - np.asarray(mgcv_fit5["start_initial"], dtype=np.float64)
                    )
                )
            )
            if mul == -1:
                start_fit = np.asarray(mgcv_fit5["start_initial"], dtype=np.float64)

                def _pen_objective_at(coef_fit):
                    ll0 = gam.family.ll(
                        gam.y_,
                        np.asarray(actual_setup.X_initial, dtype=np.float64),
                        [np.asarray(j, dtype=int) for j in actual_setup.jj],
                        np.asarray(coef_fit, dtype=np.float64),
                        np.asarray(gam.prior_weights_, dtype=np.float64),
                        offset=run["offset_list"],
                        deriv=0,
                    )["l"]
                    pen = 0.5 * float(
                        np.asarray(coef_fit, dtype=np.float64)
                        @ (np.asarray(fit["St_full"], dtype=np.float64) @ np.asarray(coef_fit, dtype=np.float64))
                    )
                    return float(ll0 - pen)

                start_d1 = gam.family.ll(
                    gam.y_,
                    np.asarray(actual_setup.X_initial, dtype=np.float64),
                    [np.asarray(j, dtype=int) for j in actual_setup.jj],
                    start_fit,
                    np.asarray(gam.prior_weights_, dtype=np.float64),
                    offset=run["offset_list"],
                    deriv=1,
                )
                analytic_pen_grad_start = np.asarray(start_d1["lb"], dtype=np.float64) - np.asarray(
                    fit["St_full"], dtype=np.float64
                ) @ start_fit
                fd_pen_grad_start = np.zeros_like(start_fit)
                eps = 1e-6
                for col in range(start_fit.size):
                    plus = start_fit.copy()
                    minus = start_fit.copy()
                    plus[col] += eps
                    minus[col] -= eps
                    fd_pen_grad_start[col] = (
                        _pen_objective_at(plus) - _pen_objective_at(minus)
                    ) / (2.0 * eps)
                row["python_pen_grad_fd_max_abs_err_at_start"] = float(
                    np.max(np.abs(analytic_pen_grad_start - fd_pen_grad_start))
                )
            report["rows"].append(row)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
