"""Canonical mgcv-style EFS outer smoothing optimizer."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from ..._model_state import (
    _coef_column_offset,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ...fit.backends import solve_fit
from ..criteria.dispatch import criterion_value
from ..criteria.ml_reml import resolve_ml_reml_scoring_backend
from ..reparam import _stable_penalty_logdet_derivatives

_EFS_LSPMAX = 15.0
_EFS_TOL = 0.1
_EFS_EPS = 1e-7


def _copy_state_vector(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def _base_penalty_matrices(model):
    n_sp = int(_n_smoothing_params(model) or 0)
    off = int(_coef_column_offset(model))
    n_full = int(_n_coef(model) + off)
    mats = [np.zeros((n_full, n_full), dtype=np.float64) for _ in range(n_sp)]
    for pb in _penalty_blocks_seq(model):
        idx = int(pb.smoothing_index)
        sl = pb.coef_slice
        full_sl = slice(off + int(sl.start), off + int(sl.stop))
        mats[idx][full_sl, full_sl] += np.asarray(pb.matrix, dtype=np.float64)
    return mats


def _expand_smoothing_params_from_log(model, log_free_sp):
    n_sp = int(_n_smoothing_params(model) or 0)
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if getattr(model, "smoothing_fixed_mask_", None) is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    log_free_sp = np.asarray(log_free_sp, dtype=np.float64).ravel()
    sp = np.asarray(
        getattr(model, "smoothing_params", np.ones(n_sp)), dtype=np.float64
    ).copy()
    sp[~fixed_mask] = np.exp(log_free_sp)
    min_sp = getattr(model, "min_sp_", None)
    if min_sp is not None:
        sp = np.maximum(sp, np.asarray(min_sp, dtype=np.float64))
    return sp


def _criterion_from_solution(model, y, sol, sp, log_sp_free, method):
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend in {"pirls_laplace", "pirls_laplace_dynamic"}:
        from ..criteria.pirls import _pirls_ml_reml_objective_from_solution

        return float(
            _pirls_ml_reml_objective_from_solution(
                model,
                np.asarray(y, dtype=np.float64),
                sol,
                np.asarray(sp, dtype=np.float64),
                str(method).upper(),
            )
        )
    if backend in {"gaussian_exact", "gaussian_dynamic"}:
        return float(
            criterion_value(
                model,
                np.asarray(y, dtype=np.float64),
                np.asarray(log_sp_free, dtype=np.float64),
                method=method,
            )
        )
    raise NotImplementedError(
        "Current EFS outer optimizer is implemented only for Gaussian and "
        "ordinary PIRLS ML/REML backends."
    )


def _trace_vs_from_solution(sol, penalty_mats):
    A_inv = np.asarray(sol.fit_state.A_inv, dtype=np.float64)
    out = np.zeros(len(penalty_mats), dtype=np.float64)
    for i, S in enumerate(penalty_mats):
        out[i] = float(np.sum(A_inv * np.asarray(S, dtype=np.float64).T))
    return out


def _b_sb_from_solution(sol, penalty_mats):
    beta = np.asarray(sol.fit_result.coef_full, dtype=np.float64).ravel()
    out = np.zeros(len(penalty_mats), dtype=np.float64)
    for i, S in enumerate(penalty_mats):
        out[i] = float(beta @ (np.asarray(S, dtype=np.float64) @ beta))
    return out


def _efs_scale(model, sol, *, edf):
    scale = float(sol.fit_result.scale)
    if (
        str(getattr(model.family, "family_class", "")).lower() == "extended"
        and getattr(model.family, "known_scale", None) is None
    ):
        n = float(model.n_samples_)
        denom = max(n - float(edf), np.finfo(np.float64).eps)
        scale = scale * n / denom
    known_scale = getattr(model.family, "known_scale", None)
    if known_scale is not None:
        scale = float(known_scale)
    return scale


def _solve_efs_step(model, y, sp, *, coef_start):
    prev = {
        "eval_coef": _copy_state_vector(getattr(model, "_pirls_eval_start_", None)),
        "eval_eta": _copy_state_vector(getattr(model, "_pirls_eval_eta_start_", None)),
        "eval_mu": _copy_state_vector(getattr(model, "_pirls_eval_mu_start_", None)),
        "lock": bool(getattr(model, "_pirls_lock_start_", False)),
        "coef": _copy_state_vector(getattr(model, "_pirls_coef_start_", None)),
        "eta": _copy_state_vector(getattr(model, "_pirls_eta_start_", None)),
        "mu": _copy_state_vector(getattr(model, "_pirls_mu_start_", None)),
    }
    try:
        model._pirls_eval_start_ = _copy_state_vector(coef_start)
        model._pirls_eval_eta_start_ = None
        model._pirls_eval_mu_start_ = None
        model._pirls_coef_start_ = _copy_state_vector(coef_start)
        model._pirls_eta_start_ = None
        model._pirls_mu_start_ = None
        model._pirls_lock_start_ = True
        fit = solve_fit(
            model, np.asarray(y, dtype=np.float64), sp, weights=model.prior_weights_
        )
        coef_out = _copy_state_vector(getattr(model, "_pirls_last_coef_", None))
    finally:
        model._pirls_eval_start_ = prev["eval_coef"]
        model._pirls_eval_eta_start_ = prev["eval_eta"]
        model._pirls_eval_mu_start_ = prev["eval_mu"]
        model._pirls_lock_start_ = prev["lock"]
        model._pirls_coef_start_ = prev["coef"]
        model._pirls_eta_start_ = prev["eta"]
        model._pirls_mu_start_ = prev["mu"]
    return fit, coef_out


def _optimize_outer_efs_mgcv(
    model,
    y,
    x0,
    bounds,
    *,
    method="reml",
    lspmax=_EFS_LSPMAX,
    efs_tol=_EFS_TOL,
):
    """Mirror `mgcv/R/gam.fit4.r::efsudr()` on ordinary families."""

    method = str(method).lower()
    if method not in {"reml", "laml"}:
        raise NotImplementedError("EFS is currently available only for REML/LAML.")

    if str(getattr(model.family, "family_class", "")).lower() == "general":
        raise NotImplementedError(
            "General-family EFS outer optimization requires a dedicated gam.fit5 "
            "mirror and is not implemented in this slice."
        )

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    x = np.asarray(x + 2.5, dtype=np.float64)
    x = np.asarray(
        _project_bounds_upper(
            x, bounds, upper=min(float(lspmax), float(max(b[1] for b in bounds)))
        ),
        dtype=np.float64,
    )
    penalty_mats = _base_penalty_matrices(model)
    n_sp = int(x.size)
    if n_sp == 0:
        return OptimizeResult(
            x=np.empty((0,), dtype=np.float64),
            fun=0.0,
            jac=np.empty((0,), dtype=np.float64),
            hess=np.empty((0, 0), dtype=np.float64),
            success=True,
            status=0,
            message="no free smoothing parameters",
            nit=0,
            nfev=0,
            njev=0,
            nhev=0,
        )

    sp = _expand_smoothing_params_from_log(model, x)
    prev_method = getattr(model, "_optim_method", None)
    model._optim_method = "REML" if method in {"reml", "laml"} else method.upper()
    try:
        fit, current_start = _solve_efs_step(
            model,
            np.asarray(y, dtype=np.float64),
            sp,
            coef_start=getattr(model, "_pirls_coef_start_", None),
        )
        score = _criterion_from_solution(model, y, fit, sp, x, method)
        mult = 1.0
        score_hist = []
        trace_rows = []
        prev_x = None
        old_dev = None
        eps_stop = float(getattr(model, "irls_tol", _EFS_EPS))
        if not np.isfinite(eps_stop) or eps_stop <= 0.0:
            eps_stop = _EFS_EPS

        for iter_idx in range(1, 201):
            iter_start = _copy_state_vector(current_start)
            bSb = _b_sb_from_solution(fit, penalty_mats)
            trVS = _trace_vs_from_solution(fit, penalty_mats)
            p = float(
                len(np.asarray(fit.fit_result.coef_full, dtype=np.float64).ravel())
            )
            edf = p - float(np.sum(trVS * np.exp(x)))
            phi = _efs_scale(model, fit, edf=edf)
            _, detS1, _ = _stable_penalty_logdet_derivatives(
                model,
                np.asarray(sp, dtype=np.float64),
                order=1,
            )
            a = np.maximum(0.0, np.asarray(detS1, dtype=np.float64) * np.exp(-x) - trVS)
            denom = np.maximum(0.0, bSb)
            r = a / denom * float(phi)
            r[(a == 0.0) & (bSb == 0.0)] = 1.0
            r[~np.isfinite(r)] = 1e6

            x1 = _project_bounds_upper(
                x + np.log(np.clip(r, 1e-300, None)) * float(mult),
                bounds,
                upper=min(float(lspmax), float(max(b[1] for b in bounds))),
            )
            max_step = float(np.max(np.abs(x1 - x))) if x.size else 0.0
            old_score = float(score)
            sp1 = _expand_smoothing_params_from_log(model, x1)
            fit1, coef1 = _solve_efs_step(
                model,
                np.asarray(y, dtype=np.float64),
                sp1,
                coef_start=iter_start,
            )
            score1 = _criterion_from_solution(model, y, fit1, sp1, x1, method)

            if score1 <= old_score:
                if max_step < 0.05:
                    x2 = _project_bounds_upper(
                        x + np.log(np.clip(r, 1e-300, None)) * float(mult) * 2.0,
                        bounds,
                        upper=min(float(lspmax), float(max(b[1] for b in bounds))),
                    )
                    sp2 = _expand_smoothing_params_from_log(model, x2)
                    fit2, coef2 = _solve_efs_step(
                        model,
                        np.asarray(y, dtype=np.float64),
                        sp2,
                        coef_start=iter_start,
                    )
                    score2 = _criterion_from_solution(model, y, fit2, sp2, x2, method)
                    if score2 < score1:
                        fit = fit2
                        current_start = _copy_state_vector(coef2)
                        score = float(score2)
                        x = np.asarray(x2, dtype=np.float64)
                        sp = np.asarray(sp2, dtype=np.float64)
                        mult = float(mult) * 2.0
                    else:
                        fit = fit1
                        current_start = _copy_state_vector(coef1)
                        score = float(score1)
                        x = np.asarray(x1, dtype=np.float64)
                        sp = np.asarray(sp1, dtype=np.float64)
                else:
                    fit = fit1
                    current_start = _copy_state_vector(coef1)
                    score = float(score1)
                    x = np.asarray(x1, dtype=np.float64)
                    sp = np.asarray(sp1, dtype=np.float64)
            else:
                while score1 > old_score and float(mult) > 1.0:
                    mult = float(mult) / 2.0
                    x1 = _project_bounds_upper(
                        x + np.log(np.clip(r, 1e-300, None)) * float(mult),
                        bounds,
                        upper=min(float(lspmax), float(max(b[1] for b in bounds))),
                    )
                    sp1 = _expand_smoothing_params_from_log(model, x1)
                    fit1, coef1 = _solve_efs_step(
                        model,
                        np.asarray(y, dtype=np.float64),
                        sp1,
                        coef_start=iter_start,
                    )
                    score1 = _criterion_from_solution(model, y, fit1, sp1, x1, method)
                fit = fit1
                current_start = _copy_state_vector(coef1)
                score = float(score1)
                x = np.asarray(x1, dtype=np.float64)
                sp = np.asarray(sp1, dtype=np.float64)
                if float(mult) < 1.0:
                    mult = 1.0

            score_hist.append(float(score))
            step_norm = (
                0.0
                if prev_x is None
                else float(np.linalg.norm(np.asarray(x, dtype=np.float64) - prev_x))
            )
            trace_rows.append(
                {
                    "iter": int(iter_idx),
                    "log_sp": np.asarray(x, dtype=np.float64).copy(),
                    "criterion": float(score),
                    "gradient": None,
                    "hessian": None,
                    "accepted_step_norm": step_norm,
                    "rank_info": {
                        "source": "outer_efs_mgcv",
                        "mult": float(mult),
                        "max_step": float(max_step),
                    },
                }
            )
            prev_x = np.asarray(x, dtype=np.float64).copy()

            dev_next = float(fit.fit_result.deviance)
            if (
                iter_idx > 3
                and max_step < 0.05
                and np.max(
                    np.abs(np.diff(np.asarray(score_hist[-4:], dtype=np.float64)))
                )
                < float(efs_tol)
            ):
                break
            if old_dev is None:
                old_dev = float(dev_next)
            else:
                if abs(float(old_dev) - float(dev_next)) < (
                    100.0 * eps_stop * abs(float(dev_next))
                ):
                    break
                old_dev = float(dev_next)
        else:
            iter_idx = 200

        model._pirls_coef_start_ = _copy_state_vector(current_start)
        model._pirls_eta_start_ = None
        model._pirls_mu_start_ = None

        result = OptimizeResult(
            x=np.asarray(x, dtype=np.float64),
            fun=float(score),
            jac=np.full(n_sp, np.nan, dtype=np.float64),
            hess=np.full((n_sp, n_sp), np.nan, dtype=np.float64),
            success=bool(iter_idx < 200),
            status=0 if iter_idx < 200 else 1,
            message="full convergence" if iter_idx < 200 else "iteration limit reached",
            nit=int(iter_idx),
            nfev=None,
            njev=0,
            nhev=0,
        )
        result.mgcv_score_hist = list(score_hist)
        result.optim_trace = trace_rows
        result.outer_info = {
            "conv": result.message,
            "iter": int(iter_idx),
            "score_hist": list(score_hist),
            "convergence": int(result.status),
            "message": str(result.message),
        }
        return result
    finally:
        model._optim_method = prev_method


def _project_bounds_upper(x, bounds, *, upper):
    x = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        hi_use = min(float(upper), float(hi))
        x[i] = min(max(float(x[i]), float(lo)), hi_use)
    return x
