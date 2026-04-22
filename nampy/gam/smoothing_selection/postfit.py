from __future__ import annotations

import re

import numpy as np
from scipy.stats import norm

from .._mgcv_constants import LINK_ETA_EXP_CLIP, LOG_GUARD_MIN
from .._model_state import (
    _fit_scale,
    _n_smoothing_params,
    _penalty_blocks_seq,
    _require_fitted,
)
from ..fit.model_ops import criterion_hessian as fit_criterion_hessian
from ..linalg import symmetrize_matrix
from .criteria.dispatch import criterion_gradient, criterion_hessian, criterion_value
from .criteria.ml_reml import resolve_ml_reml_scoring_backend


def _free_smoothing_mask(model) -> np.ndarray:
    n_sp = int(_n_smoothing_params(model) or 0)
    fixed = getattr(model, "smoothing_fixed_mask_", None)
    if fixed is None:
        return np.ones(n_sp, dtype=bool)
    return ~np.asarray(fixed, dtype=bool)


def _free_log_smoothing_params(model) -> np.ndarray:
    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    free_mask = _free_smoothing_mask(model)
    return np.log(np.clip(sp[free_mask], LOG_GUARD_MIN, None))


def _mgcv_penalty_rescale_factors(model) -> np.ndarray:
    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return np.empty((0,), dtype=np.float64)

    factors = np.ones(n_sp, dtype=np.float64)
    seen = np.zeros(n_sp, dtype=bool)

    for pb in _penalty_blocks_seq(model):
        idx = int(getattr(pb, "smoothing_index", -1))
        if idx < 0 or idx >= n_sp:
            continue
        meta = dict(getattr(pb, "metadata", {}) or {})
        s_scale = meta.get("mgcv_s_scale", None)
        if s_scale is None and (
            bool(meta.get("is_selection_penalty", False))
            or bool(getattr(pb, "is_null_space_penalty", False))
        ):
            s_scale = 1.0
        if s_scale is None:
            sid = getattr(pb, "smoothing_id", None)
            raise NotImplementedError(
                "gam_vcomp(rescale=True) missing exact mgcv penalty rescale metadata "
                f"for {sid if sid is not None else f'sp_{idx}'}."
            )

        s_scale = float(s_scale)
        if not np.isfinite(s_scale) or s_scale <= 0.0:
            raise ValueError(
                f"Invalid mgcv penalty rescale factor {s_scale!r} for smoothing "
                f"parameter index {idx}."
            )

        if seen[idx]:
            if not np.isclose(
                factors[idx], s_scale, rtol=1e-12, atol=1e-12 * max(1.0, abs(s_scale))
            ):
                raise NotImplementedError(
                    "gam_vcomp(rescale=True) requires one exact mgcv penalty "
                    f"rescale factor per smoothing parameter; index {idx} has "
                    f"{factors[idx]} and {s_scale}."
                )
        else:
            factors[idx] = s_scale
            seen[idx] = True

    if np.any(~seen):
        missing_idx = ", ".join(str(i) for i in np.flatnonzero(~seen))
        raise NotImplementedError(
            "gam_vcomp(rescale=True) missing penalty metadata for smoothing "
            f"parameter indices {missing_idx}."
        )

    return factors


def _normalize_vcomp_label(label) -> str | None:
    if label is None:
        return None
    text = str(label)
    text = re.sub(r",\s*bs\s*=\s*(\"[^\"]*\"|'[^']*'|[^,)]+)", "", text)
    text = re.sub(r",\s*k\s*=\s*[^,)]+", "", text)
    text = re.sub(
        r"^([a-zA-Z0-9_]+\([^)]*?)(?:,\s*by\s*=\s*([^)]+))\)$",
        lambda m: f"{m.group(1)}):{m.group(2).strip()}",
        text,
    )
    return text


def _vcomp_name_payload(names: list[str]):
    if len(names) == 0:
        return []
    if len(names) == 1:
        return names[0]
    return names


def _gam_vcomp_names(model) -> list[str]:
    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return []

    names: list[str | None] = [None] * n_sp
    for pb in _penalty_blocks_seq(model):
        idx = int(getattr(pb, "smoothing_index", -1))
        if idx < 0 or idx >= n_sp or names[idx] is not None:
            continue
        meta = dict(getattr(pb, "metadata", {}) or {})
        label = meta.get("formula_term", None)
        if label is None:
            label = meta.get("label", None)
        if label is None:
            label = getattr(pb, "label", None)
        names[idx] = _normalize_vcomp_label(label)

    return [
        name if name is not None else f"sp_{i}"
        for i, name in enumerate(names)
    ]


def _postfit_hessian(model, method: str, *, edge_correct: bool) -> np.ndarray | None:
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    result = getattr(model, "_optim_result", None)
    H = None if result is None else getattr(result, "hess", None)
    if edge_correct and result is not None:
        H_edge = getattr(result, "hess1", None)
        if H_edge is None:
            outer_info = getattr(result, "outer_info", {}) or {}
            H_edge = outer_info.get("hess1", None)
        if H_edge is not None:
            H = H_edge
    if backend in {"pirls_laplace", "pirls_laplace_dynamic"}:
        H = None
    if H is not None:
        H = np.asarray(H, dtype=np.float64)

    if H is None:
        H = np.asarray(
            fit_criterion_hessian(
                model,
                model.y_,
                _free_log_smoothing_params(model),
                method=method,
            ),
            dtype=np.float64,
        )
    return H


def _joint_gaussian_outer_hessian(
    model, *, edge_correct: bool
) -> np.ndarray | None:
    result = getattr(model, "_optim_result", None)
    if result is None or not bool(getattr(result, "joint_gaussian_reml_outer", False)):
        return None

    joint_x = getattr(result, "joint_x", None)
    if joint_x is None:
        return None
    joint_x = np.asarray(joint_x, dtype=np.float64).ravel()
    if joint_x.size == 0:
        return None

    outer_info = dict(getattr(result, "outer_info", {}) or {})
    H = None
    if edge_correct:
        H = outer_info.get("hess1", None)
    if H is None:
        H = outer_info.get("hess", None)
    if H is None:
        return None

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (joint_x.size, joint_x.size):
        return None
    return H


def sp_vcov(model, edge_correct: bool = True, reg: float = 1e-3):
    _require_fitted(model)

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None

    H = _joint_gaussian_outer_hessian(model, edge_correct=edge_correct)
    if H is None:
        H = _postfit_hessian(model, method, edge_correct=edge_correct)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Smoothing Hessian must be square.")
    if H.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    eye = np.eye(H.shape[0], dtype=np.float64)
    return np.linalg.solve(H + float(reg), eye)


def gam_vcomp(model, *, rescale: bool = False, conf_lev: float = 0.95):
    _require_fitted(model)

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    if sp.size == 0:
        return None
    if rescale:
        sp = sp / _mgcv_penalty_rescale_factors(model)

    scale = float(_fit_scale(model))
    vc = scale / sp
    sd = np.sqrt(vc)
    names = _gam_vcomp_names(model)
    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return {
            "vc": sd,
            "names": _vcomp_name_payload(names),
            "all": sd[0] if sd.size == 1 else sd.copy(),
            "all_names": None,
            "rank": None,
            "rank_hess": None,
            "conf_lev": None,
        }

    H = _postfit_hessian(model, method, edge_correct=False)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {"vc": sd, "names": _vcomp_name_payload(names)}
    if H.shape[0] == 0:
        return {"vc": sd, "names": _vcomp_name_payload(names)}

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
    all_names = None if sd.size == 1 else _vcomp_name_payload(names)
    return {
        "vc": ci,
        "names": _vcomp_name_payload([names[i] for i in free_idx]),
        "rank": rank,
        "rank_hess": int(H.shape[0]),
        "conf_lev": float(conf_lev),
        "all": sd,
        "all_names": all_names,
    }


def one_se_rule(model, candidate_indices: list[int] | None = None) -> np.ndarray:
    _require_fitted(model)

    V = sp_vcov(model, edge_correct=False)
    if V is None:
        raise RuntimeError("one_se_rule requires ML/REML/LAML smoothing covariance.")
    V = np.asarray(V, dtype=np.float64)

    free_idx = np.flatnonzero(_free_smoothing_mask(model))
    sp = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    log_sp_free = np.log(np.clip(sp[free_idx], LOG_GUARD_MIN, None))
    joint_gaussian = bool(
        getattr(getattr(model, "_optim_result", None), "joint_gaussian_reml_outer", False)
    )

    if candidate_indices is None:
        if joint_gaussian and V.shape[0] > free_idx.size:
            d = np.sqrt(np.clip(np.diag(V), 0.0, None))
            if np.any(d <= 0.0):
                raise RuntimeError(
                    "one_se_rule requires positive smoothing-parameter standard errors."
                )
            alpha = float(np.sqrt(2.0 * len(d)) / (d @ np.linalg.solve(V, d)))
            return np.exp(np.resize(log_sp_free, V.shape[0]) + alpha * d)

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

    log_sp = log_sp_free.copy()
    log_sp[sub_idx] = log_sp[sub_idx] + step
    sp[free_idx] = np.exp(log_sp)
    return sp


def optimizer_endpoint_diagnostics(
    model, *, conv_tol: float = 1e-6, fd_step: float = 1e-3
):
    _require_fitted(model)

    method = str(getattr(model, "_optim_method", "") or "").lower()
    n_sp = int(_n_smoothing_params(model) or 0)
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
        hess_sym = symmetrize_matrix(hess)
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
        "joint_negbin_efs_outer": bool(
            result is not None and getattr(result, "joint_negbin_efs_outer", False)
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
