"""Targeted probes for current reparameterization/postfit parity failures.

Run from repo root:
    python debug/investigate_reparam_postfit.py --case fs
    python debug/investigate_reparam_postfit.py --case gevlss
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from nampy.gam.fit.solvers.general_family_solver import build_general_family_setup_state

from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.fit.selection.optimize.basics import (
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.fit.selection.reparam import build_estimate_gam_setup_state
from tests.families.test_general_family_mgcv_parity import (
    _gevlss_data,
    _gevlss_two_smooth_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_gam_vcomp,
    _run_mgcv_snapshot,
)
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    _run_mgcv_general_preoptimization,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _compile_optimization_state,
    _run_mgcv_initial_spg,
    _run_mgcv_outer_trace,
)
from tests.optimization.test_mgcv_preoptimization_blocks_parity import (
    _make_fs_data,
    _run_mgcv_preoptimization,
)


def _arr(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).tolist()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _case_fs() -> dict:
    data = _make_fs_data()
    formula = 'y ~ s(f, x, bs="fs", k=6)'
    mgcv = _run_mgcv_preoptimization(data, formula, "gaussian", "REML")
    fit_sp_raw = mgcv.get("fit_sp", mgcv.get("sp", None))
    fit_sp = np.asarray(fit_sp_raw, dtype=np.float64).ravel()
    out = {
        "mgcv_keys": sorted(mgcv.keys()),
        "mgcv_fit_sp": fit_sp.tolist(),
        "mgcv_n_fit_sp": int(fit_sp.size),
        "mgcv_smooth": mgcv.get("smooth", None),
    }
    try:
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", fit_sp)
        out["nampy_fit_error"] = None
    except Exception as exc:  # noqa: BLE001 - debug probe records exact failure.
        out["nampy_fit_error"] = repr(exc)
        gam = _fit_nampy_model(data, formula, "gaussian", "fixed")
    cm = gam.gam_result_.compiled_model
    out["nampy_n_smoothing_params"] = int(cm.n_smoothing_params)
    out["nampy_smoothing_params"] = _arr(gam.smoothing_params)
    out["nampy_penalties"] = [
        {
            "smoothing_index": int(pb.smoothing_index),
            "smoothing_id": pb.smoothing_id,
            "kind": pb.kind,
            "rank": None if pb.rank is None else int(pb.rank),
            "shape": list(np.asarray(pb.matrix).shape),
        }
        for pb in cm.compiled_penalties
    ]
    return out


def _case_gevlss(*, two_cr: bool = False) -> dict:
    data = _gevlss_two_smooth_data() if two_cr else _gevlss_data()
    formula = (
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1"]
        if two_cr
        else ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"]
    )
    family = "gevlss"
    method = "ML"
    mgcv_vc = _run_mgcv_gam_vcomp(data, formula, family, method, rescale=False)
    mgcv_snapshot = _run_mgcv_snapshot(data, formula, family, method)
    mgcv_outer_trace = _run_mgcv_outer_trace(
        data,
        str(formula),
        family,
        method,
        "newton",
    )
    mgcv_initial = _run_mgcv_initial_spg(data, formula, family, method)
    mgcv_preopt = _run_mgcv_general_preoptimization(data, formula, family, method)
    gam_init = _compile_optimization_state(data, formula, family, method)
    nampy_initial_sp = _initial_smoothing_params_mgcv_style(gam_init, gam_init.y_)
    n_sp_init = int(np.asarray(gam_init.smoothing_params, dtype=np.float64).size)
    fit5_setup_init = build_general_family_setup_state(
        gam_init,
        np.ones(n_sp_init, dtype=np.float64),
        score_type=method,
    )
    exact_setup_init = build_estimate_gam_setup_state(gam_init)
    weights_init = (
        np.ones_like(np.asarray(gam_init.y_, dtype=np.float64).ravel())
        if gam_init.prior_weights_ is None
        else np.asarray(gam_init.prior_weights_, dtype=np.float64).ravel()
    )
    start_init = np.asarray(
        gam_init.family.initialize(
            gam_init.y_,
            fit5_setup_init.X_initial,
            fit5_setup_init.jj,
            offset=fit5_setup_init.offset_list,
            weights=weights_init,
            E=exact_setup_init.Eb,
        ),
        dtype=np.float64,
    )
    lbb_init = np.asarray(
        gam_init.family.ll(
            gam_init.y_,
            fit5_setup_init.X_initial,
            fit5_setup_init.jj,
            start_init,
            weights_init,
            offset=fit5_setup_init.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )
    mgcv_start = np.asarray(mgcv_initial.get("start", []), dtype=np.float64)
    mgcv_lbb = np.asarray(mgcv_initial.get("lbb", []), dtype=np.float64)
    mgcv_eb = np.asarray(mgcv_initial.get("Eb", []), dtype=np.float64)
    mgcv_x_initial = np.asarray(mgcv_initial.get("X_initial", []), dtype=np.float64)
    mgcv_preopt_x_initial = np.asarray(
        mgcv_preopt.get("X_initial", []), dtype=np.float64
    )
    gam = _fit_nampy_model(data, formula, family, method)
    result = getattr(gam, "_optim_result", None)
    outer_info = {} if result is None else dict(getattr(result, "outer_info", {}) or {})
    H = outer_info.get("hess", None)
    H_arr = None if H is None else np.asarray(H, dtype=np.float64)
    mgcv_sp = np.asarray(
        mgcv_snapshot["fit"].get("smoothing_params", None), dtype=np.float64
    ).ravel()
    nampy_sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()

    def _surface_at(sp):
        log_sp = np.log(np.asarray(sp, dtype=np.float64).ravel())
        return {
            "log_sp": log_sp.tolist(),
            "value": float(criterion_value(gam, gam.y_, log_sp, method=method.lower())),
            "grad": _arr(
                criterion_gradient(gam, gam.y_, log_sp, method=method.lower())
            ),
            "hess": _arr(criterion_hessian(gam, gam.y_, log_sp, method=method.lower())),
        }

    out = {
        "mgcv_gam_vcomp": mgcv_vc,
        "mgcv_fit_keys": sorted(mgcv_snapshot.get("fit", {}).keys()),
        "mgcv_sp": mgcv_sp,
        "mgcv_score": mgcv_snapshot["fit"].get("criterion_value", None),
        "mgcv_score_hist": mgcv_snapshot["fit"].get("score_hist", None),
        "mgcv_log_sp_hist": mgcv_snapshot["fit"].get("log_sp_hist", None),
        "mgcv_outer_info": mgcv_snapshot["fit"].get("outer_info", None),
        "mgcv_outer_trace": mgcv_outer_trace,
        "mgcv_initial_sp": mgcv_initial.get("initial_sp", None),
        "nampy_initial_sp": _arr(nampy_initial_sp),
        "initial_start_max_abs_diff": None
        if mgcv_start.shape != start_init.shape
        else float(np.max(np.abs(mgcv_start - start_init))),
        "initial_lbb_max_abs_diff": None
        if mgcv_lbb.shape != lbb_init.shape
        else float(np.max(np.abs(mgcv_lbb - lbb_init))),
        "initial_X_max_abs_diff": None
        if mgcv_x_initial.shape != np.asarray(fit5_setup_init.X_initial).shape
        else float(
            np.max(
                np.abs(mgcv_x_initial - np.asarray(fit5_setup_init.X_initial))
            )
        ),
        "mgcv_initial_vs_preopt_X_max_abs_diff": None
        if mgcv_x_initial.shape != mgcv_preopt_x_initial.shape
        else float(np.max(np.abs(mgcv_x_initial - mgcv_preopt_x_initial))),
        "initial_Eb_max_abs_diff": None
        if mgcv_eb.shape != exact_setup_init.Eb.shape
        else float(np.max(np.abs(mgcv_eb - exact_setup_init.Eb))),
        "initial_lbb_diag": np.diag(lbb_init).tolist(),
        "mgcv_initial_lbb_diag": np.diag(mgcv_lbb).tolist()
        if mgcv_lbb.ndim == 2
        else None,
        "mgcv_full_sp": mgcv_snapshot["fit"].get("full_sp", None),
        "mgcv_outer_grad": mgcv_snapshot["fit"].get("outer_grad", None),
        "mgcv_outer_hess": mgcv_snapshot["fit"].get("outer_hess", None),
        "mgcv_scale": mgcv_snapshot["fit"].get("scale", None),
        "nampy_gam_vcomp": _jsonable(gam.gam_vcomp(rescale=False)),
        "nampy_sp": nampy_sp,
        "nampy_scale": float(gam.gam_result_.fit_core_solution.fit_result.scale),
        "nampy_score": float(gam.smoothing_score_),
        "nampy_optim_result": {
            "success": None if result is None else bool(getattr(result, "success", False)),
            "status": None if result is None else getattr(result, "status", None),
            "message": None if result is None else str(getattr(result, "message", "")),
            "nit": None if result is None else getattr(result, "nit", None),
            "fun": None if result is None else getattr(result, "fun", None),
            "x": None if result is None else _arr(getattr(result, "x", None)),
            "jac": None if result is None else _arr(getattr(result, "jac", None)),
            "hess": None if result is None else _arr(getattr(result, "hess", None)),
            "score_hist": None
            if result is None
            else _jsonable(getattr(result, "mgcv_score_hist", None)),
        },
        "nampy_trace": _jsonable(getattr(gam, "_optim_trace", None)),
        "nampy_outer_grad": _arr(outer_info.get("grad", None)),
        "nampy_hess_shape": None if H_arr is None else list(H_arr.shape),
        "nampy_hess_eig": None
        if H_arr is None
        else np.linalg.eigvalsh(0.5 * (H_arr + H_arr.T)).tolist(),
        "nampy_surface_at_mgcv_sp": _surface_at(mgcv_sp),
        "nampy_surface_at_nampy_sp": _surface_at(nampy_sp),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("fs", "gevlss", "gevlss_two_cr"), required=True)
    args = parser.parse_args()
    if args.case == "fs":
        payload = _case_fs()
    elif args.case == "gevlss_two_cr":
        payload = _case_gevlss(two_cr=True)
    else:
        payload = _case_gevlss()
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
