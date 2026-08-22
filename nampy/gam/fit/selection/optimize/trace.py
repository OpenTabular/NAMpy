"""Optimizer trace normalization onto the model.

Pure bookkeeping extracted from the outer driver: converts optimizer- and
objective-level trace rows (joint log-scale / log-theta coordinate splits)
into the canonical ``model._optim_trace`` row format. No mgcv numerics.
"""

from __future__ import annotations

import numpy as np


def _assemble_optim_trace(model, result, objective, optimizer):
    """Normalize optimizer trace rows onto the model (extracted block)."""
    if getattr(result, "optim_trace", None) is not None and not bool(
        getattr(result, "joint_gamma_reml_outer", False)
    ):
        trace_rows = []
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        uses_joint_log_theta = bool(getattr(objective, "uses_joint_log_theta", False))
        joint_log_theta_first = bool(getattr(objective, "joint_log_theta_first", False))
        for row in list(getattr(result, "optim_trace", []) or []):
            row_dict = dict(row)
            log_sp_full = np.asarray(
                row_dict.get("log_sp", []), dtype=np.float64
            ).ravel()
            log_sp = log_sp_full.copy()
            log_scale = row_dict.get("log_scale", None)
            if log_scale is not None:
                log_scale = float(log_scale)
            log_theta = row_dict.get("log_theta", None)
            gradient_full = (
                None
                if row_dict.get("gradient", None) is None
                else np.asarray(row_dict.get("gradient"), dtype=np.float64).ravel()
            )
            hessian_full = (
                None
                if row_dict.get("hessian", None) is None
                else np.asarray(row_dict.get("hessian"), dtype=np.float64)
            )
            gradient = gradient_full
            hessian = hessian_full
            if uses_joint_log_scale and log_sp.size > 0:
                log_scale = float(log_sp[-1])
                log_sp = log_sp[:-1]
                if gradient is not None and gradient.size > 0:
                    gradient = gradient[:-1]
                if hessian is not None and hessian.shape[0] > 0:
                    hessian = hessian[:-1, :-1]
            if uses_joint_log_theta and log_sp.size > 0:
                if joint_log_theta_first:
                    log_theta = float(log_sp[0])
                    log_sp = log_sp[1:]
                    if gradient is not None and gradient.size > 0:
                        gradient = gradient[1:]
                    if hessian is not None and hessian.shape[0] > 0:
                        hessian = hessian[1:, 1:]
                else:
                    log_theta = float(log_sp[-1])
                    log_sp = log_sp[:-1]
                    if gradient is not None and gradient.size > 0:
                        gradient = gradient[:-1]
                    if hessian is not None and hessian.shape[0] > 0:
                        hessian = hessian[:-1, :-1]
            trace_rows.append(
                {
                    "iter": int(row_dict.get("iter", 0)),
                    "log_sp": log_sp.tolist(),
                    "log_scale": log_scale,
                    "log_theta": (None if log_theta is None else float(log_theta)),
                    "criterion": (
                        None
                        if row_dict.get("criterion", None) is None
                        else float(row_dict.get("criterion"))
                    ),
                    "gradient": (
                        None
                        if gradient is None
                        else np.asarray(gradient, dtype=np.float64).tolist()
                    ),
                    "gradient_full": (
                        None
                        if gradient_full is None
                        else np.asarray(gradient_full, dtype=np.float64).tolist()
                    ),
                    "hessian": (
                        None
                        if hessian is None
                        else np.asarray(hessian, dtype=np.float64).tolist()
                    ),
                    "hessian_full": (
                        None
                        if hessian_full is None
                        else np.asarray(hessian_full, dtype=np.float64).tolist()
                    ),
                    "accepted_step_norm": float(
                        row_dict.get("accepted_step_norm", 0.0)
                    ),
                    "n_fun": row_dict.get("n_fun", None),
                    "n_jac": row_dict.get("n_jac", None),
                    "n_hess": row_dict.get("n_hess", None),
                    "rank_info": row_dict.get("rank_info", None),
                }
            )
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows
    if (
        getattr(result, "optim_trace", None) is None
        and bool(getattr(result, "joint_negbin_reml_outer", False))
        and bool(getattr(result, "joint_negbin_efs_outer", False))
    ):
        outer_info = getattr(result, "outer_info", {}) or {}
        score_hist = list(outer_info.get("score_hist", []))
        log_theta_hist = list(outer_info.get("log_theta_hist", []))
        log_sp_hist = list(outer_info.get("log_sp_hist", []))
        trace_rows = []
        prev_x = None
        n_rows = min(len(score_hist), len(log_theta_hist), len(log_sp_hist))
        for i in range(n_rows):
            x_row = np.asarray(log_sp_hist[i], dtype=np.float64)
            step_norm = (
                0.0 if prev_x is None else float(np.linalg.norm(x_row - prev_x, ord=2))
            )
            trace_rows.append(
                {
                    "iter": int(i + 1),
                    "log_sp": x_row.tolist(),
                    "log_theta": float(log_theta_hist[i]),
                    "criterion": float(score_hist[i]),
                    "gradient": None,
                    "gradient_full": None,
                    "hessian": None,
                    "hessian_full": None,
                    "accepted_step_norm": step_norm,
                    "n_fun": None,
                    "n_jac": None,
                    "n_hess": None,
                    "rank_info": {
                        "joint_negbin_reml_outer": True,
                    },
                }
            )
            prev_x = x_row
        if trace_rows:
            model._optim_trace = trace_rows
            result.optim_trace = trace_rows
    elif (
        getattr(result, "optim_trace", None) is None
        and getattr(objective, "trace", None) is not None
        and (
            not bool(getattr(result, "joint_negbin_reml_outer", False))
            or not bool(getattr(result, "joint_negbin_efs_outer", False))
        )
        and not bool(getattr(result, "joint_gamma_reml_outer", False))
    ):
        trace_rows = []
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        uses_joint_log_theta = bool(getattr(objective, "uses_joint_log_theta", False))
        joint_log_theta_first = bool(getattr(objective, "joint_log_theta_first", False))
        prev_x = None
        prev_n_fun = 0
        prev_n_jac = 0
        for i, row in enumerate(objective.trace):
            x_row_full = np.asarray(row["x"], dtype=np.float64)
            x_row = x_row_full.copy()
            log_scale = None
            log_theta = None
            gradient_full = (
                None
                if row["grad"] is None
                else np.asarray(row["grad"], dtype=np.float64)
            )
            hessian_full = (
                None
                if row["hess"] is None
                else np.asarray(row["hess"], dtype=np.float64)
            )
            gradient = gradient_full
            hessian = hessian_full
            if uses_joint_log_scale and x_row.size > 0:
                log_scale = float(x_row[-1])
                x_row = x_row[:-1]
                if gradient is not None and gradient.size > 0:
                    gradient = gradient[:-1]
                if hessian is not None and hessian.shape[0] > 0:
                    hessian = hessian[:-1, :-1]
            if uses_joint_log_theta and x_row.size > 0:
                if joint_log_theta_first:
                    log_theta = float(x_row[0])
                    x_row = x_row[1:]
                    if gradient is not None and gradient.size > 0:
                        gradient = gradient[1:]
                    if hessian is not None and hessian.shape[0] > 0:
                        hessian = hessian[1:, 1:]
                else:
                    log_theta = float(x_row[-1])
                    x_row = x_row[:-1]
                    if gradient is not None and gradient.size > 0:
                        gradient = gradient[:-1]
                    if hessian is not None and hessian.shape[0] > 0:
                        hessian = hessian[:-1, :-1]
            step_norm = (
                0.0 if prev_x is None else float(np.linalg.norm(x_row - prev_x, ord=2))
            )
            n_fun = int(row.get("n_fun", 0))
            n_jac = int(row.get("n_jac", 0))
            n_hess = int(row.get("n_hess", 0))
            rank_info = None
            if optimizer == "optim":
                rank_info = {
                    "source": "outer_optim_strict",
                    "n_fun": max(0, n_fun - prev_n_fun),
                    "n_jac": max(0, n_jac - prev_n_jac),
                }
            trace_rows.append(
                {
                    "iter": int(i),
                    "log_sp": x_row.tolist(),
                    "log_scale": log_scale,
                    "log_theta": log_theta,
                    "criterion": None if row["fun"] is None else float(row["fun"]),
                    "gradient": (
                        None
                        if gradient is None
                        else np.asarray(gradient, dtype=np.float64).tolist()
                    ),
                    "gradient_full": (
                        None
                        if gradient_full is None
                        else np.asarray(gradient_full, dtype=np.float64).tolist()
                    ),
                    "hessian": (
                        None
                        if hessian is None
                        else np.asarray(hessian, dtype=np.float64).tolist()
                    ),
                    "hessian_full": (
                        None
                        if hessian_full is None
                        else np.asarray(hessian_full, dtype=np.float64).tolist()
                    ),
                    "accepted_step_norm": step_norm,
                    "n_fun": n_fun,
                    "n_jac": n_jac,
                    "n_hess": n_hess,
                    "rank_info": rank_info,
                }
            )
            prev_x = x_row
            prev_n_fun = n_fun
            prev_n_jac = n_jac
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows
    elif (
        getattr(result, "optim_trace", None) is None
        and getattr(objective, "accepted_trace", None) is not None
        and (
            not bool(getattr(result, "joint_negbin_reml_outer", False))
            or not bool(getattr(result, "joint_negbin_efs_outer", False))
        )
        and not bool(getattr(result, "joint_gamma_reml_outer", False))
    ):
        trace_rows = []
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        uses_joint_log_theta = bool(getattr(objective, "uses_joint_log_theta", False))
        joint_log_theta_first = bool(getattr(objective, "joint_log_theta_first", False))
        prev_x = None
        for i, row in enumerate(objective.accepted_trace):
            x_row_full = np.asarray(row["x"], dtype=np.float64)
            log_scale = None
            log_theta = None
            x_row = x_row_full
            if uses_joint_log_scale and x_row.size > 0:
                log_scale = float(x_row[-1])
                x_row = np.asarray(x_row[:-1], dtype=np.float64)
            if uses_joint_log_theta and x_row.size > 0:
                if joint_log_theta_first:
                    log_theta = float(x_row[0])
                    x_row = np.asarray(x_row[1:], dtype=np.float64)
                else:
                    log_theta = float(x_row[-1])
                    x_row = np.asarray(x_row[:-1], dtype=np.float64)
            step_norm = float(row.get("accepted_step_norm", 0.0))
            if prev_x is not None and not np.isfinite(step_norm):
                step_norm = float(np.linalg.norm(x_row - prev_x, ord=2))
            trace_rows.append(
                {
                    "iter": int(i + 1),
                    "log_sp": x_row.tolist(),
                    "log_scale": log_scale,
                    "log_theta": log_theta,
                    "criterion": None if row.get("fun") is None else float(row["fun"]),
                    "gradient": None,
                    "gradient_full": None,
                    "hessian": None,
                    "hessian_full": None,
                    "accepted_step_norm": step_norm,
                    "n_fun": row.get("n_fun", None),
                    "n_jac": row.get("n_jac", None),
                    "n_hess": row.get("n_hess", None),
                    "rank_info": {"accepted_outer_step": True},
                }
            )
            prev_x = x_row
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows


__all__ = ["_assemble_optim_trace"]
