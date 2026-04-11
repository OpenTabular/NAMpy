"""Rollback and acceptance heuristics for smoothing-parameter optimization."""

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from ...criteria import dispatch as _criteria_dispatch
from ..basics import _project_to_bounds


def _preserve_optimize_result_metadata(src, dst):
    """Copy non-core OptimizeResult fields that downstream diagnostics rely on."""
    if src is None or dst is None:
        return dst

    core_keys = {
        "x",
        "fun",
        "jac",
        "hess",
        "success",
        "status",
        "message",
        "nit",
        "nfev",
        "njev",
        "nhev",
    }
    for key, value in dict(src).items():
        if key in core_keys or key in dst:
            continue
        dst[key] = value
    return dst


def _criterion_infinite_sp_signal(model, y, log_sp, *, method="reml"):
    """
    Compute gradient-signal and dvkk for the infinite-smoothing rollback path.

    This avoids importing the `nampy.gam.smoothing_selection.optimize` facade at call time
    (prevents optimizer/criteria import cycles), while still allowing tests to
    monkeypatch the underlying criterion implementation.
    """
    # Access via module attribute so tests can monkeypatch the criterion implementation.
    return _criteria_dispatch.criterion_infinite_sp_signal(
        model, y, log_sp, method=method
    )


def _rollback_working_infinite_smoothing_params(
    objective,
    result,
    x0,
    bounds,
    method,
    *,
    conv_tol=1e-6,
    max_iter=5,
):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        return result

    if getattr(result, "_rolled_retry", False):
        return result
    x = np.asarray(result.x, dtype=np.float64).copy()
    if x.size == 0:
        return result

    score = float(objective.fun(x))
    grad_signal, dvkk = _criterion_infinite_sp_signal(
        objective.model, objective.y, x, method=method
    )
    grad = np.asarray(grad_signal, dtype=np.float64)
    if grad.ndim != 1 or grad.shape[0] != x.size:
        return result

    score_scale = 1.0 + abs(score)
    informative = np.abs(grad) > score_scale * conv_tol * 0.1
    if dvkk.shape == grad.shape:
        informative |= np.abs(dvkk) > score_scale * conv_tol * 0.1
    if np.all(informative):
        return result

    x_roll = x.copy()
    informative0 = informative.copy()
    rolled = False

    shrink_counts = np.zeros_like(x_roll, dtype=int)
    for _nit in range(1, int(max_iter) + 1):
        stuck = ~informative
        if not np.any(stuck):
            break
        next_counts = shrink_counts.copy()
        next_counts[stuck] += 1
        if np.any(next_counts > 1):
            break
        trial = x_roll.copy()
        trial[stuck] = 0.8 * trial[stuck] + 0.2 * x0[stuck]
        trial = _project_to_bounds(trial, bounds)
        if np.array_equal(trial, x_roll):
            break

        trial_score = float(objective.fun(trial))
        trial_grad, trial_dvkk = _criterion_infinite_sp_signal(
            objective.model, objective.y, trial, method=method
        )
        trial_grad = np.asarray(trial_grad, dtype=np.float64)
        if trial_grad.ndim != 1 or trial_grad.shape[0] != trial.size:
            break

        x_roll = trial
        score = trial_score
        grad = trial_grad
        dvkk = (
            np.asarray(trial_dvkk, dtype=np.float64)
            if np.asarray(trial_dvkk).shape == trial_grad.shape
            else np.full_like(trial_grad, np.nan)
        )

        informative = np.abs(grad) > score_scale * conv_tol * 20.0
        informative |= np.abs(dvkk) > score_scale * conv_tol * 20.0
        informative |= informative0
        rolled = True
        shrink_counts = next_counts

    if not rolled:
        return result

    rolled_result = OptimizeResult()
    rolled_result.x = x_roll.copy()
    rolled_result.fun = float(score)
    rolled_result.jac = np.asarray(objective.jac(x_roll), dtype=np.float64)
    rolled_result.hess = np.asarray(objective.hess(x_roll), dtype=np.float64)
    rolled_result.success = bool(getattr(result, "success", True))
    rolled_result.status = int(getattr(result, "status", 0))
    rolled_result.message = str(getattr(result, "message", "rolled-back endpoint"))
    rolled_result.nit = int(getattr(result, "nit", 0))
    rolled_result.nfev = int(getattr(result, "nfev", getattr(objective, "n_fun", 0)))
    rolled_result.njev = int(getattr(result, "njev", getattr(objective, "n_jac", 0)))
    rolled_result.nhev = int(getattr(result, "nhev", getattr(objective, "n_hess", 0)))
    rolled_result.rolled_back_infinite_sp = True
    rolled_result.rollback_start_x = x.copy()
    rolled_result.rollback_final_x = x_roll.copy()
    _preserve_optimize_result_metadata(result, rolled_result)

    stuck0 = ~informative0

    # Retry once from rolled point, but guard against drifting back to
    # effectively infinite smoothing on weakly identified dimensions.
    retry = minimize(
        fun=objective.fun,
        x0=x_roll,
        method="L-BFGS-B",
        jac=objective.jac if objective.use_gradient else None,
        bounds=bounds,
    )
    retry._rolled_retry = True
    retry.rolled_back_infinite_sp = True
    retry.rollback_start_x = x.copy()
    retry.rollback_final_x = x_roll.copy()
    _preserve_optimize_result_metadata(result, retry)
    if not hasattr(retry, "hess"):
        retry.hess = np.asarray(objective.hess(retry.x), dtype=np.float64)

    retry_x = np.asarray(retry.x, dtype=np.float64)
    bad_retry = False
    if retry_x.shape != x_roll.shape:
        bad_retry = True
    else:
        # If previously stuck dimensions inflate again, keep the rolled point.
        if np.any(retry_x[stuck0] > (x_roll[stuck0] + 0.5)):
            bad_retry = True
        # Also reject retry if objective meaningfully worsens.
        if float(retry.fun) > float(rolled_result.fun) + 1e-8 * (
            1.0 + abs(float(rolled_result.fun))
        ):
            bad_retry = True

    if bad_retry:
        rolled_result.retry_rejected = True
        return rolled_result

    retry.retry_rejected = False
    return retry


def _accept_flat_boundary_result(objective, result, method, *, conv_tol=1e-6):
    if bool(getattr(result, "success", False)):
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    try:
        score = float(objective.fun(x))
        grad_signal, dvkk = _criterion_infinite_sp_signal(
            objective.model, objective.y, x, method=method
        )
    except Exception:
        return result

    grad = np.asarray(grad_signal, dtype=np.float64).ravel()
    dvkk = np.asarray(dvkk, dtype=np.float64).ravel()
    if grad.shape != x.shape:
        return result
    if dvkk.shape != x.shape:
        dvkk = np.full_like(grad, np.nan)

    score_scale = 1.0 + abs(score)
    flat_boundary = np.all(np.abs(dvkk) <= score_scale * conv_tol * 1e-3)
    finite_boundary_signal = np.all(np.isfinite(grad))

    if not (flat_boundary and finite_boundary_signal):
        # L-BFGS-B can occasionally return ABNORMAL after the infinite-sp rollback
        # even when we're effectively stationary at a finite, stable point.
        # Treat that case as converged (common practical tolerance).
        message = str(getattr(result, "message", ""))
        if (
            bool(getattr(result, "rolled_back_infinite_sp", False))
            and "ABNORMAL" in message
        ):
            jac_vec = np.asarray(getattr(result, "jac", grad), dtype=np.float64).ravel()
            if jac_vec.shape == x.shape and np.all(np.isfinite(jac_vec)):
                jac_inf = float(np.linalg.norm(jac_vec, ord=np.inf))
                if jac_inf <= score_scale * conv_tol * 50.0:
                    result.success = True
                    result.message = "Accepted stationary rollback solution after L-BFGS-B abnormal termination."
                    result.fun = score
                    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
                    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
                    result.flat_boundary_accepted = True
        return result

    result.success = True
    result.message = (
        "Accepted flat boundary solution after optimizer abnormal termination."
    )
    result.fun = score
    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
    result.flat_boundary_accepted = True
    return result


def _accept_tiny_step_line_search_result(objective, result, *, step_tol=1e-7):
    if bool(getattr(result, "success", False)):
        return result

    message = str(getattr(result, "message", "")).lower()
    if "line search failed" not in message:
        return result

    trace = getattr(objective, "trace", None)
    if not trace:
        return result

    if len(trace) >= 2:
        x_last = np.asarray(trace[-1].get("x", ()), dtype=np.float64).ravel()
        x_prev = np.asarray(trace[-2].get("x", ()), dtype=np.float64).ravel()
        if x_last.shape != x_prev.shape:
            return result
        last_step = float(np.linalg.norm(x_last - x_prev, ord=2))
    else:
        last_step = 0.0

    if not np.isfinite(last_step) or last_step > float(step_tol):
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    jac = np.asarray(getattr(result, "jac", objective.jac(x)), dtype=np.float64).ravel()
    if jac.shape != x.shape or not np.all(np.isfinite(jac)):
        return result

    result.success = True
    result.status = 0
    result.message = "Accepted tiny-step line-search endpoint."
    result.fun = float(objective.fun(x))
    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
    result.tiny_step_line_search_accepted = True
    return result


def _accept_stationary_abnormal_result(
    objective,
    result,
    *,
    grad_tol=1e-6,
):
    if bool(getattr(result, "success", False)):
        return result

    message = str(getattr(result, "message", ""))
    if "ABNORMAL" not in message.upper():
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return result

    score = float(objective.fun(x))
    if not np.isfinite(score):
        return result

    jac = np.asarray(getattr(result, "jac", objective.jac(x)), dtype=np.float64).ravel()
    if jac.shape != x.shape or not np.all(np.isfinite(jac)):
        return result

    score_scale = 1.0 + abs(score)
    jac_inf = float(np.linalg.norm(jac, ord=np.inf))
    if jac_inf > score_scale * float(grad_tol):
        return result

    result.success = True
    result.status = 0
    result.message = "Accepted stationary endpoint after L-BFGS-B abnormal termination."
    result.fun = score
    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
    result.stationary_abnormal_accepted = True
    return result
