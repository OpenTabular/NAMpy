"""Gaussian performance-iteration smoothing optimizer.

This is a dedicated score/Hessian iteration corresponding to mgcv's
``magic`` identity.  It operates on the penalized least-squares criterion and
does not delegate to the generic outer Newton or BFGS implementations.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult


def optimize_magic_performance(
    objective,
    x0,
    *,
    tol=1e-7,
    step_half=15,
    max_iter=200,
    rank_tol=np.sqrt(np.finfo(float).eps),
):
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    score = float(objective.fun(x))
    trace = []
    success = False
    message = "iteration limit reached"
    for iteration in range(1, int(max_iter) + 1):
        grad = np.asarray(objective.jac(x), dtype=np.float64)
        hess = np.asarray(objective.hess(x), dtype=np.float64)
        hess = 0.5 * (hess + hess.T)
        if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            message = "non-finite magic derivative state"
            break
        if grad.size == 0 or np.linalg.norm(grad, ord=np.inf) <= float(tol) * (
            1.0 + abs(score)
        ):
            success = True
            message = "full convergence"
            break

        eigenvalues, eigenvectors = np.linalg.eigh(hess)
        floor = max(float(rank_tol), np.max(np.abs(eigenvalues)) * float(rank_tol))
        inverse_values = 1.0 / np.maximum(eigenvalues, floor)
        step = -(eigenvectors @ (inverse_values * (eigenvectors.T @ grad)))
        max_abs = float(np.max(np.abs(step))) if step.size else 0.0
        if max_abs > 5.0:
            step *= 5.0 / max_abs

        accepted = False
        alpha = 1.0
        for _ in range(int(step_half) + 1):
            trial = x + alpha * step
            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score < score:
                accepted = True
                break
            alpha *= 0.5
        trace.append(
            {
                "iter": int(iteration),
                "log_sp": x.copy(),
                "criterion": float(score),
                "gradient": grad.copy(),
                "hessian": hess.copy(),
                "accepted_step_norm": float(np.linalg.norm(alpha * step)),
                "rank_info": {"source": "magic_performance"},
            }
        )
        if not accepted:
            message = "magic step halving failed"
            break
        old_score = score
        x = trial
        score = trial_score
        if abs(old_score - score) <= float(tol) * (1.0 + abs(score)):
            success = True
            message = "full convergence"
            break

    # ``mgcv/src/magic.c`` finishes by checking each smoothing coordinate for
    # an optimum at working infinity.  It takes at most five log-SP steps of
    # length two in the direction indicated by the final gradient, retaining
    # every strict score improvement.  This is essential for flat UBRE tails
    # and is deliberately part of the magic identity rather than a generic
    # outer-optimizer boundary heuristic.
    grad = np.asarray(objective.jac(x), dtype=np.float64)
    for coordinate in range(x.size):
        direction = 1.0 if grad[coordinate] < 0.0 else -1.0
        for _ in range(5):
            trial = x.copy()
            trial[coordinate] += 2.0 * direction
            trial_score = float(objective.fun(trial))
            if not np.isfinite(trial_score) or trial_score >= score:
                break
            trace.append(
                {
                    "iter": int(iteration),
                    "log_sp": trial.copy(),
                    "criterion": float(trial_score),
                    "gradient": None,
                    "hessian": None,
                    "accepted_step_norm": 2.0,
                    "rank_info": {"source": "magic_infinity_check"},
                }
            )
            x = trial
            score = trial_score

    grad = np.asarray(objective.jac(x), dtype=np.float64)
    hess = np.asarray(objective.hess(x), dtype=np.float64)
    result = OptimizeResult(
        x=x,
        fun=float(score),
        jac=grad,
        hess=0.5 * (hess + hess.T),
        success=success,
        status=0 if success else 1,
        message=message,
        nit=int(iteration),
        nfev=getattr(objective, "n_fun", None),
        njev=getattr(objective, "n_jac", None),
        nhev=getattr(objective, "n_hess", None),
    )
    result.optim_trace = trace
    result.magic_performance_iteration = True
    result.outer_info = {
        "optimizer": "magic",
        "conv": message,
        "iter": int(iteration),
        "score_hist": [float(row["criterion"]) for row in trace],
        "gradient": grad.copy(),
        "gradient_full": grad.copy(),
        "hessian": result.hess.copy(),
        "hessian_full": result.hess.copy(),
    }
    return result


__all__ = ["optimize_magic_performance"]
