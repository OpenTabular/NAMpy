"""Canonical mgcv-style EFS outer smoothing optimizer."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from ...._mgcv_constants import LOG_GUARD_MIN
from ...._model_state import (
    _coef_column_offset,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ....linalg.cholesky import compute_preconditioned_inverse
from ...backends import solve_fit
from ...solvers.general_family.fixed_smoothing import (
    run_general_family_fixed_smoothing,
)
from ...solvers.general_family.newton import (
    _sl_block_n_params,
    _sl_repara,
    _sl_term_mult,
)
from ..criteria.gaussian_dyn import criterion_ml_reml_gaussian_dynamic_joint
from ..criteria.ml_reml import resolve_ml_reml_scoring_backend
from ..reparam import _stable_penalty_logdet_derivatives
from .basics import _initial_gaussian_scale_as_sp

_EFS_LSPMAX = 15.0
_EFS_TOL = 0.1
_EFS_EPS = 1e-7


def _copy_state_vector(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def _is_general_family(model):
    return str(getattr(model.family, "family_class", "")).lower() == "general"


def _free_smoothing_mask(model):
    n_sp = int(_n_smoothing_params(model) or 0)
    fixed_mask = getattr(model, "smoothing_fixed_mask_", None)
    if fixed_mask is None:
        return np.ones(n_sp, dtype=bool)
    return ~np.asarray(fixed_mask, dtype=bool)


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


def _criterion_from_solution(
    model, y, sol, sp, log_sp_free, method, *, gaussian_log_scale=None
):
    if _is_general_family(model):
        fit = sol["fit"]
        if method in {"reml", "laml"}:
            return float(fit["REML"])
        if method == "ml":
            return float(fit["score"])
        raise NotImplementedError(
            "General-family EFS outer optimizer is implemented only for "
            "REML/LAML in this slice."
        )
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
        if gaussian_log_scale is None:
            raise RuntimeError(
                "Gaussian EFS REML scoring requires the explicit log-scale state."
            )
        return float(
            criterion_ml_reml_gaussian_dynamic_joint(
                model,
                np.asarray(y, dtype=np.float64),
                np.asarray(log_sp_free, dtype=np.float64),
                float(gaussian_log_scale),
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


def _general_family_term_active_indices(sl_blocks):
    indices = []
    for block in sl_blocks:
        base_ind = np.arange(int(block.start) - 1, int(block.stop), dtype=int)
        if not bool(getattr(block, "linear", True)):
            for _ in range(_sl_block_n_params(block)):
                indices.append(base_ind.copy())
            continue
        active = (
            base_ind[np.asarray(block.ind, dtype=bool)]
            if bool(getattr(block, "repara", False))
            else base_ind
        )
        for _ in range(len(getattr(block, "S", ()))):
            indices.append(np.asarray(active, dtype=int).copy())
    return indices


def _general_family_vb_from_run(run):
    fit = run["fit"]
    D = np.asarray(fit["D"], dtype=np.float64)
    piv = np.asarray(fit["piv"], dtype=int)
    ipiv = np.asarray(fit["ipiv"], dtype=int)
    p = int(piv.size)
    Vb = compute_preconditioned_inverse(
        fit["L"],
        D[:p],
        p,
        piv=piv,
        ipiv=ipiv,
    )
    bdrop = np.asarray(fit["bdrop"], dtype=bool)
    if np.any(bdrop):
        q = int(bdrop.size)
        keep = ~bdrop
        Vb_full = np.zeros((q, q), dtype=np.float64)
        Vb_full[np.ix_(keep, keep)] = np.asarray(Vb, dtype=np.float64)
        Vb = Vb_full
    rp = fit.get("rp", None)
    if rp is not None:
        Vb = np.asarray(_sl_repara(rp, Vb, inverse=True), dtype=np.float64)
    return np.asarray(Vb, dtype=np.float64)


def _general_family_trace_vs_from_run(run):
    setup = run["setup"]
    Vb = _general_family_vb_from_run(run)
    SVb = _sl_term_mult(setup.Sl, Vb, full=False)
    term_indices = _general_family_term_active_indices(setup.Sl)
    out = np.zeros(len(SVb), dtype=np.float64)
    for i, (SVb_i, ind) in enumerate(zip(SVb, term_indices, strict=True)):
        out[i] = float(np.trace(np.asarray(SVb_i, dtype=np.float64)[:, ind]))
    return out


def _general_family_b_sb_from_run(run):
    setup = run["setup"]
    coef = np.asarray(run["fit"]["coef"], dtype=np.float64).ravel()
    Sb = _sl_term_mult(setup.Sl, coef, full=True)
    out = np.zeros(len(Sb), dtype=np.float64)
    for i, Sb_i in enumerate(Sb):
        out[i] = float(np.sum(coef * np.asarray(Sb_i, dtype=np.float64).ravel()))
    return out


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
        if _is_general_family(model):
            fit = run_general_family_fixed_smoothing(
                model,
                np.asarray(y, dtype=np.float64),
                sp,
                weights=model.prior_weights_,
                deriv=0,
                score_type=getattr(model, "_optim_method", "REML"),
            )
            coef_out = _copy_state_vector(fit["fit"].get("coef", None))
        else:
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


def _optimize_outer_efs_strict(
    model,
    y,
    x0,
    bounds,
    *,
    method="reml",
    lspmax=_EFS_LSPMAX,
    efs_tol=_EFS_TOL,
):
    """Mirror `mgcv/R/gam.fit4.r::efsudr()` / `efsud()`."""

    method = str(method).lower()
    if method not in {"reml", "laml"}:
        raise NotImplementedError("EFS is currently available only for REML/LAML.")

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    x = np.asarray(x + 2.5, dtype=np.float64)
    free_mask = _free_smoothing_mask(model)
    free_idx = np.flatnonzero(free_mask)
    max_bound = float(max(b[1] for b in bounds))
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
    scoring_backend = resolve_ml_reml_scoring_backend(model, method=method)
    gaussian_scale_estimated = (
        scoring_backend in {"gaussian_exact", "gaussian_dynamic"}
        and getattr(model.family, "known_scale", None) is None
    )
    if scoring_backend in {"gaussian_exact", "gaussian_dynamic"}:
        initial_scale = (
            _initial_gaussian_scale_as_sp(model, y)
            if gaussian_scale_estimated
            else float(model.family.known_scale)
        )
        gaussian_log_scale = float(np.log(max(initial_scale, LOG_GUARD_MIN)))
    else:
        gaussian_log_scale = None
    prev_method = getattr(model, "_optim_method", None)
    model._optim_method = "REML" if method in {"reml", "laml"} else method.upper()
    try:
        fit, current_start = _solve_efs_step(
            model,
            np.asarray(y, dtype=np.float64),
            sp,
            coef_start=getattr(model, "_pirls_coef_start_", None),
        )
        score = _criterion_from_solution(
            model,
            y,
            fit,
            sp,
            x,
            method,
            gaussian_log_scale=gaussian_log_scale,
        )
        if gaussian_scale_estimated:
            gaussian_log_scale = float(
                np.log(max(float(fit.fit_result.scale), LOG_GUARD_MIN))
            )
        mult = 1.0
        score_hist = []
        trace_rows = []
        prev_trace_state = None
        old_dev = None
        eps_stop = float(getattr(model, "irls_tol", _EFS_EPS))
        if not np.isfinite(eps_stop) or eps_stop <= 0.0:
            eps_stop = _EFS_EPS

        for iter_idx in range(1, 201):
            iter_start = _copy_state_vector(current_start)
            if _is_general_family(model):
                setup = fit["setup"]
                bSb_full = _general_family_b_sb_from_run(fit)
                trVS_full = _general_family_trace_vs_from_run(fit)
                detS1_full = np.asarray(setup.ldetS1, dtype=np.float64)
                bSb = np.asarray(bSb_full[free_idx], dtype=np.float64)
                trVS = np.asarray(trVS_full[free_idx], dtype=np.float64)
                detS1 = np.asarray(detS1_full[free_idx], dtype=np.float64)
                raw_a = detS1 * np.exp(-x) - trVS
                a = np.maximum(np.sqrt(np.finfo(np.float64).eps), raw_a)
                r = a / np.maximum(np.sqrt(np.finfo(np.float64).eps), bSb)
                r[(raw_a == 0.0) & (bSb == 0.0)] = 1.0
            else:
                bSb_full = _b_sb_from_solution(fit, penalty_mats)
                trVS_full = _trace_vs_from_solution(fit, penalty_mats)
                bSb = np.asarray(bSb_full[free_idx], dtype=np.float64)
                trVS = np.asarray(trVS_full[free_idx], dtype=np.float64)
                p = float(
                    len(np.asarray(fit.fit_result.coef_full, dtype=np.float64).ravel())
                )
                edf = p - float(np.sum(trVS_full * np.asarray(sp, dtype=np.float64)))
                phi = _efs_scale(model, fit, edf=edf)
                _, detS1, _ = _stable_penalty_logdet_derivatives(
                    model,
                    np.asarray(sp, dtype=np.float64),
                    order=1,
                )
                detS1 = np.asarray(detS1, dtype=np.float64)[free_idx]
                raw_a = detS1 * np.exp(-x) - trVS
                a = np.maximum(0.0, raw_a)
                denom = np.maximum(0.0, bSb)
                r = a / denom * float(phi)
                r[(a == 0.0) & (bSb == 0.0)] = 1.0
            r[~np.isfinite(r)] = 1e6

            x1 = _project_bounds_upper(
                x + np.log(np.clip(r, 1e-300, None)) * float(mult),
                bounds,
                upper=min(float(lspmax), max_bound),
            )
            max_step = float(np.max(np.abs(x1 - x))) if x.size else 0.0
            old_score = float(score)
            criterion_log_scale = gaussian_log_scale
            sp1 = _expand_smoothing_params_from_log(model, x1)
            fit1, coef1 = _solve_efs_step(
                model,
                np.asarray(y, dtype=np.float64),
                sp1,
                coef_start=iter_start,
            )
            score1 = _criterion_from_solution(
                model,
                y,
                fit1,
                sp1,
                x1,
                method,
                gaussian_log_scale=criterion_log_scale,
            )

            if score1 <= old_score:
                if max_step < 0.05:
                    x2 = _project_bounds_upper(
                        x + np.log(np.clip(r, 1e-300, None)) * float(mult) * 2.0,
                        bounds,
                        upper=(
                            min(12.0, max_bound)
                            if _is_general_family(model)
                            else min(float(lspmax), max_bound)
                        ),
                    )
                    sp2 = _expand_smoothing_params_from_log(model, x2)
                    fit2, coef2 = _solve_efs_step(
                        model,
                        np.asarray(y, dtype=np.float64),
                        sp2,
                        coef_start=iter_start,
                    )
                    score2 = _criterion_from_solution(
                        model,
                        y,
                        fit2,
                        sp2,
                        x2,
                        method,
                        gaussian_log_scale=criterion_log_scale,
                    )
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
                        upper=min(float(lspmax), max_bound),
                    )
                    sp1 = _expand_smoothing_params_from_log(model, x1)
                    fit1, coef1 = _solve_efs_step(
                        model,
                        np.asarray(y, dtype=np.float64),
                        sp1,
                        coef_start=iter_start,
                    )
                    score1 = _criterion_from_solution(
                        model,
                        y,
                        fit1,
                        sp1,
                        x1,
                        method,
                        gaussian_log_scale=criterion_log_scale,
                    )
                fit = fit1
                current_start = _copy_state_vector(coef1)
                score = float(score1)
                x = np.asarray(x1, dtype=np.float64)
                sp = np.asarray(sp1, dtype=np.float64)
                if float(mult) < 1.0:
                    mult = 1.0

            if gaussian_scale_estimated:
                # `mgcv::efsudr()` carries the scale estimate from `fit1` into
                # the next iteration.  This remains `fit1` even when the
                # optional doubled step (`fit2`) is accepted.
                gaussian_log_scale = float(
                    np.log(max(float(fit1.fit_result.scale), LOG_GUARD_MIN))
                )

            score_hist.append(float(score))
            trace_state = np.asarray(x, dtype=np.float64)
            if gaussian_log_scale is not None:
                trace_state = np.concatenate(
                    [trace_state, np.array([gaussian_log_scale], dtype=np.float64)]
                )
            step_norm = (
                0.0
                if prev_trace_state is None
                else float(np.linalg.norm(trace_state - prev_trace_state))
            )
            trace_rows.append(
                {
                    "iter": int(iter_idx),
                    "log_sp": np.asarray(x, dtype=np.float64).copy(),
                    "log_scale": gaussian_log_scale,
                    "criterion": float(score),
                    "gradient": None,
                    "hessian": None,
                    "accepted_step_norm": step_norm,
                    "rank_info": {
                        "source": "outer_efs_strict",
                        "mult": float(mult),
                        "max_step": float(max_step),
                    },
                }
            )
            prev_trace_state = trace_state.copy()

            dev_next = (
                float(fit["fit"]["l"])
                if _is_general_family(model)
                else float(fit.fit_result.deviance)
            )
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
        result.strict_score_hist = list(score_hist)
        result.optim_trace = trace_rows
        result.outer_info = {
            "optimizer": "efs",
            "conv": result.message,
            "iter": int(iter_idx),
            "score_hist": list(score_hist),
            "log_scale": None,
            "log_theta": None,
            "gradient": None,
            "gradient_full": None,
            "hessian": None,
            "hessian_full": None,
            "edge_correct": False,
            "lsp1": None,
            "hess1": None,
            "convergence": int(result.status),
            "message": str(result.message),
            "counts": None,
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
