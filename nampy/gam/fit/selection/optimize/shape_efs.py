"""Direct SCAM ``efsudr.scam2`` GCV/UBRE smoothing iteration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from ....model_state import _fit_workspace


def _derivatives(objective, x):
    objective.jac(x)
    state = _fit_workspace(objective.model).get("transformed_gcv_ubre_state", None)
    if not isinstance(state, dict):
        raise RuntimeError("SCAM EFS requires transformed criterion derivative state.")
    return (
        np.asarray(state["deviance_gradient"], dtype=np.float64),
        np.asarray(state["trace_gradient"], dtype=np.float64),
        state["solution"],
    )


def optimize_shape_efs(objective, x0, *, lspmax=15.0, efs_tol=0.1, max_iter=200):
    """Mirror ``scam/R/estimate.scam.R::efsudr.scam2``."""
    x = np.asarray(x0, dtype=np.float64).ravel().copy() + 2.5
    x = np.minimum(x, float(lspmax))
    mult = 1.0
    score = float(objective.fun(x))
    score_hist = []
    trace = []
    old_dev = None
    converged = False
    max_iter = int(max_iter)
    for iteration in range(1, max_iter + 1):
        dev1, tau1, solution = _derivatives(objective, x)
        dev = float(solution["deviance"])
        tr_a = float(solution["trace_H"])
        n = float(objective.model.n_samples_)
        gamma = float(objective.model.score_gamma)
        scale = getattr(objective.model.family, "known_scale", None)
        if scale is None:
            a = np.maximum(0.0, -2.0 * gamma * dev * tau1 / (n - tr_a))
        else:
            a = np.maximum(0.0, -2.0 * gamma * float(scale) * tau1)
        denom = np.maximum(0.0, dev1)
        ratio = a / denom
        ratio[(a == 0.0) | (dev1 == 0.0)] = 1.0
        ratio[~np.isfinite(ratio)] = 1e6
        direction = np.log(np.clip(ratio, 1e-300, None))
        trial_x = np.minimum(x + direction * mult, float(lspmax))
        max_step = float(np.max(np.abs(trial_x - x))) if x.size else 0.0
        old_score = score
        trial_score = float(objective.fun(trial_x))
        if trial_score <= old_score:
            if max_step < 0.05:
                extended_x = np.minimum(x + direction * mult * 2.0, float(lspmax))
                extended_score = float(objective.fun(extended_x))
                if extended_score < trial_score:
                    trial_x, trial_score = extended_x, extended_score
                    mult *= 2.0
            x, score = trial_x, trial_score
        else:
            threshold = 10.0 * (0.1 + abs(old_score)) * np.sqrt(np.finfo(float).eps)
            halves = 0
            while (
                not np.isfinite(trial_score) or trial_score - old_score > threshold
            ) and halves < 15:
                mult *= 0.5
                trial_x = np.minimum(x + direction * mult, float(lspmax))
                trial_score = float(objective.fun(trial_x))
                halves += 1
            x, score = trial_x, trial_score
            if mult < 1.0:
                mult = 1.0
        score_hist.append(float(score))
        trace.append(
            {
                "iter": int(iteration),
                "log_sp": x.copy(),
                "criterion": float(score),
                "gradient": None,
                "hessian": None,
                "accepted_step_norm": float(max_step),
                "rank_info": {"source": "shape_efs", "mult": float(mult)},
            }
        )
        current_state = _fit_workspace(objective.model).get(
            "transformed_gcv_ubre_state", None
        )
        current_dev = float(current_state["solution"]["deviance"])
        if (
            iteration > 3
            and max_step < 0.05
            and np.max(np.abs(np.diff(score_hist[-4:]))) < float(efs_tol)
        ):
            converged = True
            break
        if old_dev is not None and abs(old_dev - current_dev) < 100.0 * float(
            objective.model.control.scam_devtol_fit
        ) * abs(current_dev):
            converged = True
            break
        old_dev = current_dev
    result = OptimizeResult(
        x=x,
        fun=float(score),
        jac=np.full(x.size, np.nan),
        hess=np.full((x.size, x.size), np.nan),
        success=converged,
        status=0 if converged else 1,
        message=("full convergence" if converged else "iteration limit reached"),
        nit=int(iteration),
        nfev=getattr(objective, "n_fun", None),
        njev=getattr(objective, "n_jac", None),
        nhev=0,
    )
    result.optim_trace = trace
    result.score_hist = np.asarray(score_hist, dtype=np.float64)
    result.shape_efs = True
    result.outer_info = {
        "optimizer": "efs",
        "iter": int(iteration),
        "conv": str(result.message),
        "score_hist": result.score_hist.copy(),
    }
    return result


__all__ = ["optimize_shape_efs"]
