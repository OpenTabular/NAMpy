"""Post-optimization adjustments (rollback, refinement, acceptance heuristics)."""
import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from .basics import _project_to_bounds
from ..criteria import dispatch as _criteria_dispatch


def _criterion_infinite_sp_signal(model, y, log_sp, *, method="reml"):
    """
    Compute gradient-signal and dvkk for the infinite-smoothing rollback path.

    This avoids importing the `nampy.gam.smoothing_selection.optimize` facade at call time
    (prevents optimizer/criteria import cycles), while still allowing tests to
    monkeypatch the underlying criterion implementation.
    """
    # Access via module attribute so tests can monkeypatch the criterion implementation.
    return _criteria_dispatch.criterion_infinite_sp_signal(model, y, log_sp, method=method)


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
    nit = 0
    for nit in range(1, int(max_iter) + 1):
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
        if float(retry.fun) > float(rolled_result.fun) + 1e-8 * (1.0 + abs(float(rolled_result.fun))):
            bad_retry = True

    if bad_retry:
        rolled_result.retry_rejected = True
        return rolled_result

    retry.retry_rejected = False
    return retry


def _stabilize_flat_smoothing_params(
    objective,
    result,
    x0,
    bounds,
    method,
    *,
    conv_tol=1e-6,
    flat_score_rel_tol=2e-7,
    log_step=0.1,
):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
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
    flat = np.abs(grad) <= score_scale * conv_tol * 0.5

    if not np.any(flat):
        return result

    improved = False
    x_work = x.copy()
    score_ref = score
    score_tol = max(1e-12, score_scale * float(flat_score_rel_tol))
    step = max(float(log_step), 1e-6)

    for j in np.flatnonzero(flat):
        local_x = x_work.copy()
        local_best = local_x[j]
        local_best_score = score_ref
        lower_bound = float(bounds[j][0])
        for _ in range(256):
            trial = local_x.copy()
            trial[j] = max(lower_bound, trial[j] - step)
            trial = _project_to_bounds(trial, bounds)
            if trial[j] >= local_x[j] - 1e-12:
                break
            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score <= local_best_score + score_tol:
                local_x = trial
                local_best = trial[j]
                local_best_score = trial_score
                if local_best <= lower_bound + 1e-12:
                    break
            else:
                break
        if local_best < x_work[j] - 1e-12:
            x_work[j] = local_best
            improved = True

    if not improved:
        return result

    score_work = float(objective.fun(x_work))
    result.x = x_work
    result.fun = float(score_work)
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.flat_sp_stabilized = True
    return result


def _collapse_near_zero_smoothing_params(objective, result, bounds, method, *, conv_tol=1e-6):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        return result

    x = np.asarray(result.x, dtype=np.float64).copy()
    if x.size == 0:
        return result

    score = float(objective.fun(x))
    grad_signal, dvkk = _criterion_infinite_sp_signal(
        objective.model, objective.y, x, method=method
    )
    grad = np.asarray(grad_signal, dtype=np.float64)
    dvkk = np.asarray(dvkk, dtype=np.float64)
    if grad.ndim != 1 or grad.shape[0] != x.size:
        return result
    if dvkk.shape != grad.shape:
        dvkk = np.full_like(grad, np.nan)

    score_scale = 1.0 + abs(score)
    wants_smaller_sp = grad > score_scale * conv_tol
    near_zero_curvature = np.abs(dvkk) <= score_scale * conv_tol * 1e-3
    collapse = wants_smaller_sp & near_zero_curvature

    if not np.any(collapse):
        return result

    improved = False
    x_work = x.copy()
    score_work = score
    score_tol = max(1e-8, score_scale * 1e-9)

    for j in np.flatnonzero(collapse):
        local_x = x_work.copy()
        local_best = local_x[j]
        local_best_score = score_work
        lower_bound = float(bounds[j][0])

        for _ in range(256):
            trial = local_x.copy()
            trial[j] = max(lower_bound, trial[j] - 2.0)
            trial = _project_to_bounds(trial, bounds)
            if trial[j] >= local_x[j] - 1e-12:
                break

            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score <= local_best_score + score_tol:
                local_x = trial
                local_best = trial[j]
                local_best_score = trial_score
                if local_best <= lower_bound + 1e-12:
                    break
            else:
                break

        if local_best < x_work[j] - 1e-12:
            x_work[j] = local_best
            score_work = local_best_score
            improved = True

    if not improved:
        return result

    result.x = x_work
    result.fun = float(score_work)
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.near_zero_sp_collapsed = True
    return result


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
        if bool(getattr(result, "rolled_back_infinite_sp", False)) and "ABNORMAL" in message:
            jac_vec = np.asarray(getattr(result, "jac", grad), dtype=np.float64).ravel()
            if jac_vec.shape == x.shape and np.all(np.isfinite(jac_vec)):
                jac_inf = float(np.linalg.norm(jac_vec, ord=np.inf))
                if jac_inf <= score_scale * conv_tol * 50.0:
                    result.success = True
                    result.message = (
                        "Accepted stationary rollback solution after L-BFGS-B abnormal termination."
                    )
                    result.fun = score
                    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
                    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
                    result.flat_boundary_accepted = True
        return result

    result.success = True
    result.message = "Accepted flat boundary solution after optimizer abnormal termination."
    result.fun = score
    result.jac = np.asarray(objective.jac(x), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x), dtype=np.float64)
    result.flat_boundary_accepted = True
    return result


def _snap_gaussian_random_effect_boundary(
    objective,
    result,
    bounds,
    method,
    *,
    snap_log_sp=-64.0,
):
    method = str(method).lower()
    if method not in {"reml", "laml"}:
        return result

    model = getattr(objective, "model", None)
    if model is None:
        return result
    if str(getattr(getattr(model, "family", None), "name", "")).lower() != "gaussian":
        return result

    term_blocks = getattr(model, "term_blocks_", None) or ()
    if not any(str(getattr(tb, "term_type", "")).lower() == "random_effect" for tb in term_blocks):
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    grad_signal, dvkk = _criterion_infinite_sp_signal(
        model, objective.y, x, method=method
    )
    grad = np.asarray(grad_signal, dtype=np.float64).ravel()
    dvkk = np.asarray(dvkk, dtype=np.float64).ravel()
    if grad.shape != x.shape:
        return result
    if dvkk.shape != x.shape:
        dvkk = np.full_like(grad, np.nan)

    score = float(objective.fun(x))
    score_scale = 1.0 + abs(score)
    wants_smaller_sp = grad > score_scale * 1e-6
    near_zero_curvature = np.abs(dvkk) <= score_scale * 1e-9
    near_boundary_ridge = wants_smaller_sp & near_zero_curvature & (x < -20.0)
    if not np.any(near_boundary_ridge):
        return result

    x_snap = x.copy()
    for j in np.flatnonzero(near_boundary_ridge):
        lo = float(bounds[j][0])
        x_snap[j] = max(lo, float(snap_log_sp))

    if np.allclose(x_snap, x, atol=0.0, rtol=0.0):
        return result

    result.x = x_snap
    result.fun = float(objective.fun(x_snap))
    result.jac = np.asarray(objective.jac(x_snap), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_snap), dtype=np.float64)
    result.gaussian_re_boundary_snapped = True
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


def _refine_null_space_smoothing_params(objective, result, bounds, *, xatol=1e-3):
    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    penalty_blocks = getattr(objective.model, "penalty_blocks_", None) or []
    null_full = sorted(
        {
            int(pb.smoothing_index)
            for pb in penalty_blocks
            if bool(getattr(pb, "is_null_space_penalty", False))
        }
    )
    if not null_full:
        return result

    fixed_mask = getattr(objective.model, "smoothing_fixed_mask_", None)
    if fixed_mask is None:
        fixed_mask = np.zeros(int(getattr(objective.model, "n_smoothing_params_", 0) or 0), dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
    free_to_full = np.flatnonzero(~fixed_mask)
    null_free = [j for j, full_idx in enumerate(free_to_full) if int(full_idx) in null_full]
    if not null_free:
        return result

    x_work = x.copy()
    improved = False

    def _set_coord(vec, j, value):
        trial = np.asarray(vec, dtype=np.float64).copy()
        trial[j] = float(value)
        return trial

    for j in null_free:
        g0 = np.asarray(objective.jac(x_work), dtype=np.float64).ravel()
        if g0.shape != x_work.shape or not np.all(np.isfinite(g0)):
            continue
        g0j = float(g0[j])
        if abs(g0j) <= 1e-6:
            continue

        lo, hi = bounds[j]
        lo = float(lo)
        hi = float(hi)
        left = right = float(x_work[j])
        step = 0.5
        bracket = None

        if g0j > 0.0:
            right = float(x_work[j])
            for _ in range(64):
                left = max(lo, right - step)
                g_left = float(np.asarray(objective.jac(_set_coord(x_work, j, left)), dtype=np.float64).ravel()[j])
                if not np.isfinite(g_left):
                    break
                if g_left <= 0.0 or left <= lo + 1e-12:
                    bracket = (left, right)
                    break
                right = left
        else:
            left = float(x_work[j])
            for _ in range(64):
                right = min(hi, left + step)
                g_right = float(
                    np.asarray(objective.jac(_set_coord(x_work, j, right)), dtype=np.float64).ravel()[j]
                )
                if not np.isfinite(g_right):
                    break
                if g_right >= 0.0 or right >= hi - 1e-12:
                    bracket = (left, right)
                    break
                left = right

        if bracket is None:
            continue

        a, b = bracket
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            continue

        def _fun_1d(v):
            return float(objective.fun(_set_coord(x_work, j, v)))

        opt = minimize_scalar(
            _fun_1d,
            bounds=(float(a), float(b)),
            method="bounded",
            options={"xatol": float(xatol)},
        )
        if not bool(getattr(opt, "success", False)) or not np.isfinite(getattr(opt, "fun", np.nan)):
            continue

        if float(opt.fun) + 1e-10 < float(objective.fun(x_work)):
            x_work[j] = float(opt.x)
            improved = True

    if not improved:
        return result

    result.x = x_work
    result.fun = float(objective.fun(x_work))
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.null_space_refined = True
    return result


def _coordinate_refine_smoothing_params(
    objective,
    result,
    bounds,
    *,
    max_passes=6,
    xatol=1e-10,
    improve_tol=1e-12,
):
    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    x_work = x.copy()
    best_fun = float(objective.fun(x_work))
    if not np.isfinite(best_fun):
        return result

    improved_any = False

    def _set_coord(vec, j, value):
        trial = np.asarray(vec, dtype=np.float64).copy()
        trial[j] = float(value)
        return trial

    for _ in range(int(max_passes)):
        improved_pass = False
        for j, (lo, hi) in enumerate(bounds):
            lo = float(lo)
            hi = float(hi)
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                continue

            def _fun_1d(v):
                return float(objective.fun(_set_coord(x_work, j, v)))

            opt = minimize_scalar(
                _fun_1d,
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": float(xatol)},
            )
            if not bool(getattr(opt, "success", False)) or not np.isfinite(
                getattr(opt, "fun", np.nan)
            ):
                continue

            if float(opt.fun) + float(improve_tol) < best_fun:
                x_work[j] = float(opt.x)
                best_fun = float(opt.fun)
                improved_pass = True
                improved_any = True

        if not improved_pass:
            break

    if not improved_any:
        return result

    result.x = x_work
    result.fun = float(best_fun)
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.coordinate_refined = True
    return result
