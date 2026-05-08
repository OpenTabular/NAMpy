"""Canonical Newton entry points for smoothing-parameter optimisation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from ..._mgcv_constants import PENALTY_RIDGE_REL
from .basics import _project_to_bounds
from .newton_strict import _optimize_outer_newton_strict


def optimize_outer_newton_generic(
    objective,
    x0,
    bounds,
    max_iter=50,
    grad_tol=1e-8,
    step_tol=1e-8,
):
    """Generic safeguarded Newton solve for non-mgcv outer criteria."""
    x = _project_to_bounds(x0, bounds)
    f = float(objective.fun(x))
    success = False
    message = "maximum iterations reached"
    nit = 0

    for _nit in range(1, max_iter + 1):
        g = np.asarray(objective.jac(x), dtype=np.float64)
        if not np.all(np.isfinite(g)):
            message = "non-finite gradient"
            break
        if np.linalg.norm(g, ord=np.inf) <= grad_tol:
            success = True
            message = "gradient tolerance satisfied"
            break

        H = np.asarray(objective.hess(x), dtype=np.float64)
        if not np.all(np.isfinite(H)):
            message = "non-finite Hessian"
            break

        direction = None
        ridge = 0.0
        eye = np.eye(len(x), dtype=np.float64)
        for _ in range(8):
            try:
                direction_try = -np.linalg.solve(H + ridge * eye, g)
            except np.linalg.LinAlgError:
                ridge = max(
                    PENALTY_RIDGE_REL,
                    10.0 * ridge if ridge > 0 else PENALTY_RIDGE_REL,
                )
                continue
            if float(g @ direction_try) < 0.0 and np.all(np.isfinite(direction_try)):
                direction = direction_try
                break
            ridge = max(
                PENALTY_RIDGE_REL,
                10.0 * ridge if ridge > 0 else PENALTY_RIDGE_REL,
            )

        if direction is None:
            direction = -g

        step_norm = float(np.linalg.norm(direction))
        if step_norm <= step_tol:
            success = True
            message = "step tolerance satisfied"
            break

        accepted = False
        candidate_directions = [direction]
        if float(g @ direction) >= 0.0:
            candidate_directions = []
        candidate_directions.append(-g)

        for cand_direction in candidate_directions:
            alpha = 1.0
            gtd = float(g @ cand_direction)
            for _ in range(25):
                x_trial = _project_to_bounds(x + alpha * cand_direction, bounds)
                if np.array_equal(x_trial, x):
                    alpha *= 0.5
                    continue
                f_trial = float(objective.fun(x_trial))
                if np.isfinite(f_trial) and f_trial <= f + 1e-4 * alpha * gtd:
                    x = x_trial
                    f = f_trial
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break

        if not accepted:
            message = "line search failed"
            break

    g_final = np.asarray(objective.jac(x), dtype=np.float64)
    h_final = np.asarray(objective.hess(x), dtype=np.float64)
    nit = _nit if max_iter > 0 else 0
    return OptimizeResult(
        x=x,
        fun=float(f),
        jac=g_final,
        hess=h_final,
        success=bool(success),
        status=0 if success else 1,
        message=message,
        nit=nit,
        nfev=objective.n_fun,
        njev=objective.n_jac,
        nhev=objective.n_hess,
    )


def optimize_outer_newton_indefinite_hessian(
    objective,
    x0,
    bounds,
    *,
    conv_tol=1e-6,
    step_tol=1e-7,
    max_iter=200,
    max_nstep=5.0,
    max_sstep=2.0,
    max_half=30,
    qerror_thresh=0.8,
    edge_correct=False,
):
    """mgcv-shaped Newton solve that tolerates indefinite outer Hessians."""
    model = getattr(objective, "model", None)
    score_type = str(getattr(objective, "method", "reml")).upper()
    prev_irls_tol = None
    if model is not None:
        prev_irls_tol = float(getattr(model, "irls_tol", 1e-7))
        if prev_irls_tol > float(conv_tol) / 100.0:
            model.irls_tol = float(conv_tol) / 100.0

    try:

        def _eval_at(
            x_eval,
            *,
            start_coef,
            start_eta=None,
            start_mu=None,
            need_grad=False,
            need_hess=False,
            commit_start=False,
        ):
            if model is not None:
                if start_coef is None:
                    start_coef = getattr(model, "_pirls_coef_start_", None)
                if start_mu is None:
                    start_mu = getattr(model, "_pirls_mu_start_", None)

            x_eval = np.asarray(x_eval, dtype=np.float64).ravel()

            if model is not None:
                model._pirls_eval_start_ = (
                    None
                    if start_coef is None
                    else np.asarray(start_coef, dtype=np.float64).copy()
                )
                # Mirror `mgcv/R/gam.fit3.r::newton()`: outer Newton carries
                # `start` and `mustart`, but does not pass `etastart` between
                # outer evaluations.
                model._pirls_eval_eta_start_ = None
                model._pirls_eval_mu_start_ = (
                    None
                    if start_mu is None
                    else np.asarray(start_mu, dtype=np.float64).copy()
                )
                # Keep the PIRLS warm start fixed across score/gradient/Hessian
                # evaluations at the same outer point. mgcv only advances
                # `start`/`mustart` after the point is accepted.
                model._pirls_lock_start_ = True

            objective._last_x = None
            objective._last_fun = None
            objective._last_grad = None
            objective._last_hess = None

            score_eval = float(objective.fun(x_eval))
            grad_eval = (
                np.asarray(objective.jac(x_eval), dtype=np.float64)
                if need_grad
                else None
            )
            hess_eval = None
            restore_general_fit5_hessian = None
            if need_hess:
                if (
                    model is not None
                    and str(
                        getattr(getattr(model, "family", None), "family_class", "")
                    ).lower()
                    == "general"
                    and str(getattr(objective, "method", "")).lower()
                    in {"ml", "reml", "laml"}
                ):
                    restore_general_fit5_hessian = bool(
                        getattr(
                            model,
                            "_general_family_outer_use_fit5_hessian_",
                            False,
                        )
                    )
                    model._general_family_outer_use_fit5_hessian_ = True
                try:
                    hess_eval = np.asarray(objective.hess(x_eval), dtype=np.float64)
                finally:
                    if restore_general_fit5_hessian is not None:
                        model._general_family_outer_use_fit5_hessian_ = (
                            restore_general_fit5_hessian
                        )
            if (
                getattr(objective, "_last_fun", None) is not None
                and hasattr(objective, "_same_x")
                and bool(objective._same_x(x_eval))
            ):
                score_eval = float(objective._last_fun)

            coef_eval = (
                getattr(model, "_pirls_last_coef_", None) if model is not None else None
            )
            eta_eval = (
                getattr(model, "_pirls_last_eta_", None) if model is not None else None
            )
            mu_eval = (
                getattr(model, "_pirls_last_mu_", None) if model is not None else None
            )
            if model is not None:
                if coef_eval is not None:
                    coef_eval = np.asarray(coef_eval, dtype=np.float64).copy()
                    if commit_start:
                        model._pirls_coef_start_ = coef_eval.copy()
                if eta_eval is not None:
                    eta_eval = np.asarray(eta_eval, dtype=np.float64).copy()
                if mu_eval is not None:
                    mu_eval = np.asarray(mu_eval, dtype=np.float64).copy()
                    if commit_start:
                        model._pirls_mu_start_ = mu_eval.copy()

            dvkk_diag = np.full(x_eval.shape, np.nan, dtype=np.float64)
            gamma_state = (
                getattr(model, "_pirls_reml_gamma_state_", None)
                if model is not None
                else None
            )
            scale_est = None

            if isinstance(gamma_state, dict):
                scale_obj = gamma_state.get("scale_est", None)
                if (
                    scale_obj is not None
                    and np.isfinite(scale_obj)
                    and float(scale_obj) > 0.0
                ):
                    scale_est = float(scale_obj)

            if (
                scale_est is None
                and model is not None
                and bool(getattr(objective, "uses_joint_log_scale", False))
                and str(getattr(getattr(model, "family", None), "name", "")).lower()
                == "gaussian"
            ):
                scale_obj = getattr(model, "_gaussian_reml_last_scale_est_", None)
                if (
                    scale_obj is not None
                    and np.isfinite(scale_obj)
                    and float(scale_obj) > 0.0
                ):
                    scale_est = float(scale_obj)

            if (
                scale_est is None
                and bool(getattr(objective, "uses_joint_log_scale", False))
                and x_eval.size > 0
            ):
                phi = float(np.exp(float(x_eval[-1])))
                if np.isfinite(phi) and phi > 0.0:
                    scale_est = phi

            if scale_est is None and isinstance(gamma_state, dict):
                phi = gamma_state.get("phi", None)
                if phi is not None and np.isfinite(phi) and float(phi) > 0.0:
                    scale_est = float(phi)

            if model is not None:
                if commit_start:
                    if coef_eval is not None:
                        model._pirls_coef_start_ = np.asarray(
                            coef_eval, dtype=np.float64
                        ).copy()
                    if mu_eval is not None:
                        model._pirls_mu_start_ = np.asarray(mu_eval, dtype=np.float64).copy()
                model._pirls_eval_start_ = None
                model._pirls_eval_eta_start_ = None
                model._pirls_eval_mu_start_ = None
                model._pirls_lock_start_ = False

            return (
                score_eval,
                grad_eval,
                hess_eval,
                dvkk_diag,
                coef_eval,
                eta_eval,
                mu_eval,
                scale_est,
            )

        return _optimize_outer_newton_strict(
            objective=objective,
            x0=x0,
            bounds=bounds,
            eval_at=_eval_at,
            score_type=score_type,
            conv_tol=conv_tol,
            max_nstep=max_nstep,
            max_sstep=max_sstep,
            max_half=max_half,
            qerror_thresh=qerror_thresh,
            max_iter=max_iter,
            step_tol=step_tol,
            edge_correct=edge_correct,
        )
    finally:
        if model is not None and prev_irls_tol is not None:
            model.irls_tol = prev_irls_tol


def optimize_outer_newton_strict(*args, **kwargs):
    """Public alias for direct mgcv-style Newton calls."""
    return _optimize_outer_newton_strict(*args, **kwargs)


_optimize_outer_newton = optimize_outer_newton_generic
_optimize_outer_newton_indefinite_hessian = optimize_outer_newton_indefinite_hessian


__all__ = [
    "optimize_outer_newton_generic",
    "optimize_outer_newton_indefinite_hessian",
    "optimize_outer_newton_strict",
    "_optimize_outer_newton",
    "_optimize_outer_newton_indefinite_hessian",
]
