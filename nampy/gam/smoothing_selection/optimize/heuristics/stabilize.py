"""Stabilization and refinement heuristics for smoothing-parameter optimization."""

import numpy as np
from scipy.optimize import minimize_scalar

from ...._mgcv_constants import FAMILY_EPS, SMOOTHING_SCORE_ABS_FLOOR
from ...._model_state import _term_blocks_seq
from ..basics import _project_to_bounds
from .rollback import _criterion_infinite_sp_signal


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


def _stabilize_joint_negbin_flat_ridge(
    objective,
    result,
    bounds,
    *,
    log_step=0.1,
    score_tol=1.0e-4,
    flat_grad_tol=5e-4,
    flat_hess_tol=5e-4,
    max_shift=2.0,
):
    if not bool(getattr(result, "joint_negbin_reml_outer", False)):
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    grad = np.asarray(objective.jac(x), dtype=np.float64).ravel()
    if grad.shape != x.shape or not np.all(np.isfinite(grad)):
        return result
    hess = np.asarray(objective.hess(x), dtype=np.float64)
    if hess.shape != (x.size, x.size) or not np.all(np.isfinite(hess)):
        return result
    hess_diag = np.abs(np.diag(hess))
    flat = (np.abs(grad) <= float(flat_grad_tol)) & (hess_diag <= float(flat_hess_tol))
    flat_idx = np.flatnonzero(flat)
    if flat_idx.size == 0:
        return result

    best_x = x.copy()
    best_score = float(objective.fun(best_x))
    if not np.isfinite(best_score):
        return result

    step = max(float(log_step), 1e-6)
    improved = False
    ref_score = best_score
    if x.size == 1:
        j = 0
        lower_bound = float(bounds[j][0])
        local_best = best_x.copy()
        while True:
            trial = local_best.copy()
            trial[j] = max(lower_bound, trial[j] - step)
            trial = _project_to_bounds(trial, bounds)
            if trial[j] >= local_best[j] - 1e-12:
                break
            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score <= ref_score + float(score_tol):
                local_best = trial
                best_x = trial
                improved = True
                continue
            break
    elif flat_idx.size > 1:
        x_work = best_x.copy()
        max_total_shift = max(float(max_shift), step)
        changed = True
        while changed:
            changed = False
            order = flat_idx[np.argsort(-x_work[flat_idx])]
            for j in order.tolist():
                lower_bound = float(bounds[j][0])
                total_shift = 0.0
                while total_shift + step <= max_total_shift + 1e-12:
                    trial = x_work.copy()
                    trial[j] = max(lower_bound, trial[j] - step)
                    trial = _project_to_bounds(trial, bounds)
                    shift = abs(float(trial[j] - x_work[j]))
                    if shift <= 1e-12:
                        break
                    trial_score = float(objective.fun(trial))
                    if not np.isfinite(trial_score) or trial_score > ref_score + float(
                        score_tol
                    ):
                        break
                    x_work = trial
                    total_shift += shift
                    improved = True
                    changed = True
        best_x = x_work
        best_score = float(objective.fun(best_x))
    else:
        j = int(flat_idx[0])
        lower_bound = float(bounds[j][0])
        local_best = best_x.copy()
        total_shift = 0.0
        max_total_shift = max(float(max_shift), step)
        while total_shift + step <= max_total_shift + 1e-12:
            trial = local_best.copy()
            trial[j] = max(lower_bound, trial[j] - step)
            trial = _project_to_bounds(trial, bounds)
            shift = abs(float(trial[j] - local_best[j]))
            if shift <= 1e-12:
                break
            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score <= ref_score + float(score_tol):
                local_best = trial
                best_x = trial
                improved = True
                total_shift += shift
                continue
            break
    if not improved:
        return result

    result.x = best_x.copy()
    result.fun = float(best_score)
    result.jac = np.asarray(objective.jac(best_x), dtype=np.float64)
    result.hess = np.asarray(objective.hess(best_x), dtype=np.float64)
    result.joint_negbin_flat_ridge_stabilized = True
    return result


def _collapse_near_zero_smoothing_params(
    objective, result, bounds, method, *, conv_tol=1e-6
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
    score_tol = max(SMOOTHING_SCORE_ABS_FLOOR, score_scale * FAMILY_EPS)

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

    term_blocks = _term_blocks_seq(model)
    if not any(
        str(getattr(tb, "term_type", "")).lower() == "random_effect"
        for tb in term_blocks
    ):
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
    near_zero_curvature = np.abs(dvkk) <= score_scale * FAMILY_EPS
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
        fixed_mask = np.zeros(
            int(getattr(objective.model, "n_smoothing_params_", 0) or 0), dtype=bool
        )
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
    free_to_full = np.flatnonzero(~fixed_mask)
    null_free = [
        j for j, full_idx in enumerate(free_to_full) if int(full_idx) in null_full
    ]
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
                g_left = float(
                    np.asarray(
                        objective.jac(_set_coord(x_work, j, left)), dtype=np.float64
                    ).ravel()[j]
                )
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
                    np.asarray(
                        objective.jac(_set_coord(x_work, j, right)), dtype=np.float64
                    ).ravel()[j]
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

        def _fun_1d(v, j=j):
            return float(objective.fun(_set_coord(x_work, j, v)))

        opt = minimize_scalar(
            _fun_1d,
            bounds=(float(a), float(b)),
            method="bounded",
            options={"xatol": float(xatol)},
        )
        if not bool(getattr(opt, "success", False)) or not np.isfinite(
            getattr(opt, "fun", np.nan)
        ):
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


def _stabilize_factor_smooth_shared_ridge(
    objective,
    result,
    bounds,
    method,
    *,
    score_tol_abs=2e-5,
    log_step=0.25,
    max_shift=4.0,
):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        return result

    model = getattr(objective, "model", None)
    if model is None:
        return result
    if str(getattr(getattr(model, "family", None), "name", "")).lower() != "gaussian":
        return result

    x = np.asarray(getattr(result, "x", ()), dtype=np.float64).ravel()
    if x.size == 0:
        return result

    fixed_mask = getattr(model, "smoothing_fixed_mask_", None)
    if fixed_mask is None:
        fixed_mask = np.zeros(
            int(getattr(model, "n_smoothing_params_", 0) or 0), dtype=bool
        )
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
    free_to_full = np.flatnonzero(~fixed_mask)
    full_to_free = {int(full): int(i) for i, full in enumerate(free_to_full)}

    fs_groups = []
    for tb in _term_blocks_seq(model):
        if str(getattr(tb, "term_type", "")).lower() != "factor_smooth_fs":
            continue
        group = sorted(
            {
                int(pb.smoothing_index)
                for pb in (getattr(model, "penalty_blocks_", None) or [])
                if pb.coef_slice == tb.coef_slice
            }
        )
        group_free = [full_to_free[g] for g in group if g in full_to_free]
        if group_free:
            fs_groups.append(group_free)

    if not fs_groups:
        return result

    base_score = float(objective.fun(x))
    if not np.isfinite(base_score):
        return result

    score_tol = max(float(score_tol_abs), (1.0 + abs(base_score)) * 1e-7)
    improved = False
    x_work = x.copy()
    score_work = base_score

    for group in fs_groups:
        local = x_work.copy()
        local_best = local.copy()
        local_best_score = score_work
        total_shift = 0.0

        while total_shift + log_step <= max_shift + 1e-12:
            trial = local.copy()
            stop = False
            for j in group:
                hi = float(bounds[j][1])
                trial[j] = min(hi, trial[j] + float(log_step))
                if trial[j] <= local[j] + 1e-12:
                    stop = True
            if stop:
                break

            trial = _project_to_bounds(trial, bounds)
            trial_score = float(objective.fun(trial))
            if not np.isfinite(trial_score) or trial_score > base_score + score_tol:
                break

            local = trial
            local_best = trial.copy()
            local_best_score = trial_score
            total_shift += float(log_step)

        if np.any(local_best[group] > x_work[group] + 1e-12):
            x_work = local_best
            score_work = local_best_score
            improved = True

    if not improved:
        return result

    result.x = x_work
    result.fun = float(score_work)
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.factor_smooth_shared_ridge_stabilized = True
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

            def _fun_1d(v, j=j):
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
