import json
from pathlib import Path

import numpy as np


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
                "log_sp": np.asarray(row.get("log_sp", []), dtype=np.float64).tolist(),
                "log_scale": (
                    None
                    if row.get("log_scale", None) is None
                    else float(row.get("log_scale"))
                ),
                "log_theta": (
                    None
                    if row.get("log_theta", None) is None
                    else float(row.get("log_theta"))
                ),
                "criterion": (
                    None
                    if row.get("criterion", None) is None
                    else float(row.get("criterion"))
                ),
                "gradient": (
                    None
                    if row.get("gradient", None) is None
                    else np.asarray(row.get("gradient"), dtype=np.float64).tolist()
                ),
                "hessian": (
                    None
                    if row.get("hessian", None) is None
                    else np.asarray(row.get("hessian"), dtype=np.float64).tolist()
                ),
                "accepted_step_norm": float(row.get("accepted_step_norm", 0.0)),
                "rank_info": row.get("rank_info", None),
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
