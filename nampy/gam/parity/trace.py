import json
from pathlib import Path

import numpy as np

_SOURCE_TO_OUTER_OPTIMIZER = {
    "mgcv_newton": "newton",
    "outer_newton_mgcv": "newton",
    "mgcv_bfgs": "bfgs",
    "outer_bfgs_mgcv": "bfgs",
    "mgcv_efs": "efs",
    "outer_efs_mgcv": "efs",
    "mgcv_optim": "optim",
}


def _normalize_trace_value(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _normalize_trace_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_trace_value(val) for val in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_trace_value(value.item())
        return [_normalize_trace_value(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_trace_value(value.item())
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _normalize_outer_info(raw_outer_info, *, optim_result, trace_rows):
    raw_outer_info = {} if raw_outer_info is None else dict(raw_outer_info)
    out = _normalize_trace_value(raw_outer_info)

    if "grad" in raw_outer_info and "gradient" not in out:
        out["gradient"] = _normalize_trace_value(raw_outer_info["grad"])
        out["gradient_full"] = _normalize_trace_value(raw_outer_info["grad"])
    if "hess" in raw_outer_info and "hessian" not in out:
        out["hessian"] = _normalize_trace_value(raw_outer_info["hess"])
        out["hessian_full"] = _normalize_trace_value(raw_outer_info["hess"])

    joint_log_scale = None
    for attr_name in ("joint_log_phi", "joint_log_sigma2"):
        attr_value = getattr(optim_result, attr_name, None)
        if attr_value is not None:
            joint_log_scale = attr_value
            break
    if "log_scale" not in out:
        out["log_scale"] = _normalize_trace_value(joint_log_scale)
    if "log_theta" not in out:
        out["log_theta"] = _normalize_trace_value(
            getattr(optim_result, "joint_log_theta", None)
        )

    if "edge_correct" not in out:
        edge_correct = getattr(optim_result, "mgcv_edge_correct_applied", None)
        if edge_correct is None:
            edge_correct = getattr(optim_result, "mgcv_edge_correct", None)
        out["edge_correct"] = _normalize_trace_value(edge_correct)

    if "lsp1" not in out:
        out["lsp1"] = _normalize_trace_value(getattr(optim_result, "lsp1", None))
    if "hess1" not in out:
        out["hess1"] = _normalize_trace_value(getattr(optim_result, "hess1", None))

    if "convergence" not in out:
        out["convergence"] = _normalize_trace_value(getattr(optim_result, "status", None))
    if "message" not in out:
        out["message"] = _normalize_trace_value(getattr(optim_result, "message", None))

    if "counts" not in out:
        counts = []
        nfev = getattr(optim_result, "nfev", None)
        njev = getattr(optim_result, "njev", None)
        if nfev is not None:
            counts.append(int(nfev))
        if njev is not None:
            counts.append(int(njev))
        out["counts"] = counts or None

    if "optimizer" not in out:
        source = None
        if trace_rows:
            rank_info = trace_rows[0].get("rank_info", None) or {}
            source = rank_info.get("source", None)
        out["optimizer"] = _SOURCE_TO_OUTER_OPTIMIZER.get(source, None)

    return out


def build_optimizer_trace(model):
    core = model
    if hasattr(model, "core_") and model.core_ is not None:
        core = model.core_
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "core_")
        and model.model.core_ is not None
    ):
        core = model.model.core_

    rows = getattr(core, "_optim_trace", None)
    if rows is None:
        rows = []

    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "iter": int(row.get("iter", 0)),
                "log_sp": _normalize_trace_value(row.get("log_sp", [])),
                "log_scale": _normalize_trace_value(row.get("log_scale", None)),
                "log_theta": _normalize_trace_value(row.get("log_theta", None)),
                "criterion": _normalize_trace_value(row.get("criterion", None)),
                "gradient": _normalize_trace_value(row.get("gradient", None)),
                "gradient_full": _normalize_trace_value(
                    row.get("gradient_full", row.get("gradient", None))
                ),
                "hessian": _normalize_trace_value(row.get("hessian", None)),
                "hessian_full": _normalize_trace_value(
                    row.get("hessian_full", row.get("hessian", None))
                ),
                "accepted_step_norm": float(row.get("accepted_step_norm", 0.0)),
                "n_fun": _normalize_trace_value(row.get("n_fun", None)),
                "n_jac": _normalize_trace_value(row.get("n_jac", None)),
                "n_hess": _normalize_trace_value(row.get("n_hess", None)),
                "rank_info": _normalize_trace_value(row.get("rank_info", None)),
            }
        )

    optim_result = getattr(core, "_optim_result", None)
    optim_success = (
        None if optim_result is None else getattr(optim_result, "success", None)
    )
    optim_nit = None if optim_result is None else getattr(optim_result, "nit", None)
    edge_correct = (
        None
        if optim_result is None
        else getattr(optim_result, "mgcv_edge_correct", None)
    )
    edge_correct_applied = (
        None
        if optim_result is None
        else getattr(optim_result, "mgcv_edge_correct_applied", None)
    )
    fit = {
        "criterion_name": getattr(core, "_optim_method", None),
        "smoothing_params": np.asarray(
            getattr(core, "smoothing_params", []), dtype=np.float64
        ).tolist(),
        "converged": None if optim_success is None else bool(optim_success),
        "message": (
            None if optim_result is None else str(getattr(optim_result, "message", ""))
        ),
        "optimizer_nit": None if optim_nit is None else int(optim_nit),
        "edge_correct": None if edge_correct is None else bool(edge_correct),
        "edge_correct_applied": (
            None if edge_correct_applied is None else bool(edge_correct_applied)
        ),
        "outer_info": (
            None
            if optim_result is None
            else _normalize_outer_info(
                getattr(optim_result, "outer_info", None),
                optim_result=optim_result,
                trace_rows=out_rows,
            )
        ),
    }
    return {"fit": fit, "trace": out_rows}


def save_optimizer_trace(trace_obj, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(trace_obj, f, indent=2)


def load_optimizer_trace(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
