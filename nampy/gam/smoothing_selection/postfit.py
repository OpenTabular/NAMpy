from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .._mgcv_constants import LINK_ETA_EXP_CLIP, LOG_GUARD_MIN
from .._model_state import _require_fitted
from .criteria.dispatch import criterion_gradient, criterion_hessian, criterion_value
from .criteria.ml_reml import resolve_ml_reml_scoring_backend


def _free_smoothing_mask(model) -> np.ndarray:
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    fixed = getattr(model, "smoothing_fixed_mask_", None)
    if fixed is None:
        return np.ones(n_sp, dtype=bool)
    return ~np.asarray(fixed, dtype=bool)


def _free_log_smoothing_params(model) -> np.ndarray:
    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    free_mask = _free_smoothing_mask(model)
    return np.log(np.clip(sp[free_mask], LOG_GUARD_MIN, None))


def sp_vcov(model, edge_correct: bool = True, reg: float = 1e-3):
    del edge_correct
    _require_fitted(model)

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None

    backend = resolve_ml_reml_scoring_backend(model, method=method)
    result = getattr(model, "_optim_result", None)
    H = None if result is None else getattr(result, "hess", None)
    if backend in {"pirls_laplace", "pirls_laplace_dynamic"}:
        H = None
    if H is None:
        log_sp = _free_log_smoothing_params(model)
        H = np.asarray(
            model._criterion_hessian(model.y_, log_sp, method=method), dtype=np.float64
        )
    else:
        H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Smoothing Hessian must be square.")
    if H.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.linalg.solve(H + float(reg), np.eye(H.shape[0], dtype=np.float64))


def gam_vcomp(model, *, rescale: bool = False, conf_lev: float = 0.95):
    _require_fitted(model)
    if rescale:
        raise NotImplementedError(
            "gam_vcomp(rescale=True) requires stored penalty rescaling factors, which are not yet retained."
        )

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    if sp.size == 0:
        return None

    scale = float(model.scale_)
    vc = scale / sp
    sd = np.sqrt(vc)
    names = [f"sp_{i}" for i in range(len(sp))]
    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return {"vc": sd, "names": names}

    result = getattr(model, "_optim_result", None)
    H = None if result is None else getattr(result, "hess", None)
    if H is None:
        log_sp = _free_log_smoothing_params(model)
        H = np.asarray(
            model._criterion_hessian(model.y_, log_sp, method=method), dtype=np.float64
        )
    else:
        H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {"vc": sd, "names": names}
    if H.shape[0] == 0:
        return {"vc": sd, "names": names}

    evals, evecs = np.linalg.eigh(H)
    keep = evals > np.max(evals) * np.finfo(np.float64).eps ** 0.75
    rank = int(np.sum(keep))
    inv_vals = np.zeros_like(evals)
    inv_vals[keep] = 1.0 / evals[keep]
    V = evecs @ (inv_vals[:, None] * evecs.T)

    crit = float(norm.ppf(1.0 - (1.0 - float(conf_lev)) / 2.0))
    lsd = np.log(sd[_free_smoothing_mask(model)])
    J = -0.5 * np.eye(V.shape[0], dtype=np.float64)
    V_lsd = J @ V @ J.T
    sd_lsd = np.sqrt(np.clip(np.diag(V_lsd), 0.0, None))
    ci = np.column_stack(
        [
            np.exp(np.clip(lsd, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP)),
            np.exp(np.clip(lsd - crit * sd_lsd, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP)),
            np.exp(np.clip(lsd + crit * sd_lsd, -LINK_ETA_EXP_CLIP, LINK_ETA_EXP_CLIP)),
        ]
    )
    free_idx = np.flatnonzero(_free_smoothing_mask(model))
    return {
        "vc": ci,
        "names": [names[i] for i in free_idx],
        "rank": rank,
        "rank_hess": int(H.shape[0]),
        "conf_lev": float(conf_lev),
        "all": sd,
        "all_names": names,
    }


def one_se_rule(model, candidate_indices: list[int] | None = None) -> np.ndarray:
    _require_fitted(model)

    V = sp_vcov(model, edge_correct=False)
    if V is None:
        raise RuntimeError("one_se_rule requires ML/REML/LAML smoothing covariance.")

    free_idx = np.flatnonzero(_free_smoothing_mask(model))
    if candidate_indices is None:
        sub_idx = np.arange(free_idx.size, dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int).ravel()
        pos = []
        for idx in candidate_indices:
            hit = np.flatnonzero(free_idx == int(idx))
            if hit.size == 0:
                continue
            pos.append(int(hit[0]))
        if len(pos) == 0:
            raise ValueError(
                "candidate_indices does not contain any estimated smoothing parameters."
            )
        sub_idx = np.asarray(pos, dtype=int)

    V_sub = np.asarray(V[np.ix_(sub_idx, sub_idx)], dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(V_sub), 0.0, None))
    if np.any(d <= 0.0):
        raise RuntimeError(
            "one_se_rule requires positive smoothing-parameter standard errors."
        )
    alpha = float(np.sqrt(2.0 * len(d)) / (d @ np.linalg.solve(V_sub, d)))
    step = alpha * d

    sp = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    log_sp = np.log(np.clip(sp[free_idx], LOG_GUARD_MIN, None))
    log_sp[sub_idx] = log_sp[sub_idx] + step
    sp[free_idx] = np.exp(log_sp)
    return sp


def optimizer_endpoint_diagnostics(
    model, *, conv_tol: float = 1e-6, fd_step: float = 1e-3
):
    _require_fitted(model)

    method = str(getattr(model, "_optim_method", "") or "").lower()
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    result = getattr(model, "_optim_result", None)
    if method in {"", "fixed"} or n_sp == 0:
        return None

    free_mask = _free_smoothing_mask(model)
    x = _free_log_smoothing_params(model)
    n_free = int(x.size)
    if n_free == 0:
        return None

    min_sp = (
        np.zeros(n_sp, dtype=np.float64)
        if getattr(model, "min_sp_", None) is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )
    bounds = []
    for lower_sp in min_sp[free_mask]:
        if lower_sp > 0:
            lo = max(float(model.sp_log_bounds[0]), float(np.log(lower_sp)))
        else:
            lo = float(model.sp_log_bounds[0])
        bounds.append((lo, float(model.sp_log_bounds[1])))
    bounds = np.asarray(bounds, dtype=np.float64)

    grad = None
    if result is not None and getattr(result, "jac", None) is not None:
        grad = np.asarray(result.jac, dtype=np.float64).ravel()
        if grad.shape != x.shape or not np.all(np.isfinite(grad)):
            grad = None
    if grad is None:
        grad = np.asarray(
            criterion_gradient(model, model.y_, x, method=method), dtype=np.float64
        ).ravel()

    hess = None
    if result is not None and getattr(result, "hess", None) is not None:
        hess = np.asarray(result.hess, dtype=np.float64)
        if hess.shape != (n_free, n_free) or not np.all(np.isfinite(hess)):
            hess = None
    if hess is None:
        try:
            hess = np.asarray(
                criterion_hessian(model, model.y_, x, method=method), dtype=np.float64
            )
            if hess.shape != (n_free, n_free) or not np.all(np.isfinite(hess)):
                hess = None
        except Exception:
            hess = None

    score = getattr(model, "smoothing_score_", None)
    if score is None or not np.isfinite(float(score)):
        score = float(criterion_value(model, model.y_, x, method=method))
    else:
        score = float(score)
    score_scale = 1.0 + abs(score)
    tol = float(conv_tol) * score_scale
    factor_smooth_shared_ridge_stabilized = False
    factor_smooth_shared_ridge_shift = None

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    at_lower = x <= (lower + 1e-10)
    at_upper = x >= (upper - 1e-10)

    projected_grad = grad.copy()
    projected_grad[at_lower] = np.minimum(projected_grad[at_lower], 0.0)
    projected_grad[at_upper] = np.maximum(projected_grad[at_upper], 0.0)

    eigvals = None
    shared_curvature = None
    min_abs_eig = None
    if hess is not None:
        hess_sym = 0.5 * (hess + hess.T)
        eigvals = np.linalg.eigvalsh(hess_sym)
        min_abs_eig = float(np.min(np.abs(eigvals))) if eigvals.size else 0.0
        u = np.full(n_free, 1.0 / np.sqrt(n_free), dtype=np.float64)
        shared_curvature = float(u @ hess_sym @ u)
    else:
        u = np.full(n_free, 1.0 / np.sqrt(n_free), dtype=np.float64)

    shared_slope = float(u @ grad)
    max_down = float(np.min((x - lower) / u)) if n_free > 0 else 0.0
    max_up = float(np.min((upper - x) / u)) if n_free > 0 else 0.0
    shared_step = min(float(fd_step), max(0.0, max_down), max(0.0, max_up))
    shared_fd_slope = None
    shared_fd_curvature = None
    if shared_step > 0.0:
        f0 = score
        fp = float(criterion_value(model, model.y_, x + shared_step * u, method=method))
        fm = float(criterion_value(model, model.y_, x - shared_step * u, method=method))
        shared_fd_slope = float((fp - fm) / (2.0 * shared_step))
        shared_fd_curvature = float((fp - 2.0 * f0 + fm) / (shared_step * shared_step))

    hess_scale = (
        float(np.max(np.abs(eigvals)))
        if eigvals is not None and eigvals.size > 0
        else 0.0
    )
    flat_ridge_suspected = bool(
        np.linalg.norm(projected_grad, ord=np.inf) <= max(tol * 5.0, 1e-8)
        and hess is not None
        and (
            (min_abs_eig is not None and min_abs_eig <= max(1e-8, hess_scale * 1e-6))
            or (
                shared_curvature is not None
                and abs(shared_curvature) <= max(1e-8, hess_scale * 1e-6)
            )
        )
    )
    return {
        "criterion_name": method,
        "criterion_backend": resolve_ml_reml_scoring_backend(model, method=method),
        "optimizer_success": (
            None if result is None else bool(getattr(result, "success", False))
        ),
        "optimizer_message": (
            None if result is None else str(getattr(result, "message", ""))
        ),
        "joint_gaussian_reml_outer": bool(
            result is not None and getattr(result, "joint_gaussian_reml_outer", False)
        ),
        "joint_negbin_reml_outer": bool(
            result is not None and getattr(result, "joint_negbin_reml_outer", False)
        ),
        "joint_negbin_postprocessed": bool(
            result is not None and getattr(result, "joint_negbin_postprocessed", False)
        ),
        "joint_negbin_flat_ridge_stabilized": bool(
            result is not None
            and getattr(result, "joint_negbin_flat_ridge_stabilized", False)
        ),
        "joint_log_theta": (
            None if result is None else getattr(result, "joint_log_theta", None)
        ),
        "joint_negbin_initial_log_theta": (
            None
            if result is None
            else getattr(result, "joint_negbin_initial_log_theta", None)
        ),
        "joint_negbin_optimizer_message": (
            None if result is None else getattr(result, "joint_negbin_message", None)
        ),
        "joint_negbin_optimizer_fun": (
            None if result is None else getattr(result, "joint_negbin_fun", None)
        ),
        "joint_negbin_optimizer_nfev": (
            None if result is None else getattr(result, "joint_negbin_nfev", None)
        ),
        "joint_negbin_optimizer_njev": (
            None if result is None else getattr(result, "joint_negbin_njev", None)
        ),
        "family_theta": (
            float(model.family.theta) if hasattr(model.family, "theta") else None
        ),
        "n_free_smoothing_params": n_free,
        "log_smoothing_params": x.tolist(),
        "bounds": bounds.tolist(),
        "gradient": grad.tolist(),
        "projected_gradient": projected_grad.tolist(),
        "gradient_inf_norm": float(np.linalg.norm(grad, ord=np.inf)),
        "projected_gradient_inf_norm": float(
            np.linalg.norm(projected_grad, ord=np.inf)
        ),
        "stationary_by_raw_gradient": bool(np.linalg.norm(grad, ord=np.inf) <= tol),
        "stationary_by_projected_gradient": bool(
            np.linalg.norm(projected_grad, ord=np.inf) <= tol
        ),
        "boundary_limited": bool(
            np.any(at_lower | at_upper)
            and np.linalg.norm(projected_grad, ord=np.inf) <= tol
            and np.linalg.norm(grad, ord=np.inf) > tol
        ),
        "at_lower_bound": at_lower.tolist(),
        "at_upper_bound": at_upper.tolist(),
        "hessian": None if hess is None else hess.tolist(),
        "hessian_eigenvalues": None if eigvals is None else eigvals.tolist(),
        "min_abs_hessian_eigenvalue": min_abs_eig,
        "shared_shift_directional_derivative": shared_slope,
        "shared_shift_curvature": shared_curvature,
        "shared_shift_fd_step": (None if shared_step <= 0.0 else float(shared_step)),
        "shared_shift_fd_slope": shared_fd_slope,
        "shared_shift_fd_curvature": shared_fd_curvature,
        "factor_smooth_shared_ridge_stabilized": factor_smooth_shared_ridge_stabilized,
        "factor_smooth_shared_ridge_shift": factor_smooth_shared_ridge_shift,
        "flat_ridge_suspected": flat_ridge_suspected,
    }


__all__ = ["sp_vcov", "gam_vcomp", "one_se_rule", "optimizer_endpoint_diagnostics"]
