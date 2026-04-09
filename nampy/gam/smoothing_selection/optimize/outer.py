"""Outer Newton optimizers for log smoothing parameters."""

import numpy as np
from scipy.optimize import OptimizeResult

from .basics import _project_to_bounds


def _optimize_outer_newton(
    objective, x0, bounds, max_iter=50, grad_tol=1e-6, step_tol=1e-8
):
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
                ridge = max(1e-6, 10.0 * ridge if ridge > 0 else 1e-6)
                continue
            if float(g @ direction_try) < 0.0 and np.all(np.isfinite(direction_try)):
                direction = direction_try
                break
            ridge = max(1e-6, 10.0 * ridge if ridge > 0 else 1e-6)

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


def _optimize_outer_newton_indefinite_hessian(
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
):
    model = objective.model
    score_type = str(getattr(objective, "method", "reml")).upper()

    def _score_scale(score_val, old_score_val, scale_est=None):
        score_val = float(score_val)
        old_score_val = float(old_score_val)
        if score_type in {"REML", "P-REML", "ML", "P-ML"}:
            if scale_est is None:
                scale_obj = getattr(model, "scale_", 1.0)
                scale = 1.0 if scale_obj is None else float(scale_obj)
            else:
                scale = float(scale_est)
            score_scale_val = abs(np.log(abs(scale))) + abs(score_val)
        else:
            score_scale_val = abs(score_val)
        if score_scale_val <= 0.0:
            if abs(score_val) < abs(old_score_val):
                score_scale_val = abs(old_score_val)
            else:
                score_scale_val = 1.0
        return float(score_scale_val)

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
        x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
        model._pirls_eval_start_ = (
            None
            if start_coef is None
            else np.asarray(start_coef, dtype=np.float64).copy()
        )
        model._pirls_eval_eta_start_ = (
            None
            if start_eta is None
            else np.asarray(start_eta, dtype=np.float64).copy()
        )
        model._pirls_eval_mu_start_ = (
            None if start_mu is None else np.asarray(start_mu, dtype=np.float64).copy()
        )
        model._pirls_lock_start_ = not bool(commit_start)
        objective._last_x = None
        objective._last_fun = None
        objective._last_grad = None
        objective._last_hess = None
        score_eval = float(objective.fun(x_eval))
        grad_eval = (
            np.asarray(objective.jac(x_eval), dtype=np.float64) if need_grad else None
        )
        hess_eval = (
            np.asarray(objective.hess(x_eval), dtype=np.float64) if need_hess else None
        )
        coef_eval = getattr(model, "_pirls_last_coef_", None)
        eta_eval = getattr(model, "_pirls_last_eta_", None)
        mu_eval = getattr(model, "_pirls_last_mu_", None)
        if coef_eval is not None:
            coef_eval = np.asarray(coef_eval, dtype=np.float64).copy()
            if commit_start:
                model._pirls_coef_start_ = coef_eval.copy()
        if eta_eval is not None:
            eta_eval = np.asarray(eta_eval, dtype=np.float64).copy()
            if commit_start:
                model._pirls_eta_start_ = eta_eval.copy()
        if mu_eval is not None:
            mu_eval = np.asarray(mu_eval, dtype=np.float64).copy()
            if commit_start:
                model._pirls_mu_start_ = mu_eval.copy()
        dvkk_diag = np.full(x_eval.shape, np.nan, dtype=np.float64)
        gamma_state = getattr(model, "_pirls_reml_gamma_state_", None)
        scale_eval = None
        if isinstance(gamma_state, dict):
            scale_obj = gamma_state.get("scale_est", None)
            if (
                scale_obj is not None
                and np.isfinite(scale_obj)
                and float(scale_obj) > 0.0
            ):
                scale_eval = float(scale_obj)
        if (
            scale_eval is None
            and bool(getattr(objective, "uses_joint_log_scale", False))
            and x_eval.size > 0
        ):
            phi = float(np.exp(float(x_eval[-1])))
            if np.isfinite(phi) and phi > 0.0:
                scale_eval = phi
        if scale_eval is None and isinstance(gamma_state, dict):
            phi = gamma_state.get("phi", None)
            if phi is not None and np.isfinite(phi) and float(phi) > 0.0:
                scale_eval = float(phi)
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
            scale_eval,
        )

    x = _project_to_bounds(np.asarray(x0, dtype=np.float64), bounds)
    accepted_start = getattr(model, "_pirls_coef_start_", None)
    accepted_eta = getattr(model, "_pirls_eta_start_", None)
    accepted_mu = getattr(model, "_pirls_mu_start_", None)
    score, grad, hess, dvkk, coef0, eta0, mu0, scale_est = _eval_at(
        x,
        start_coef=accepted_start,
        start_eta=accepted_eta,
        start_mu=accepted_mu,
        need_grad=True,
        need_hess=True,
        commit_start=True,
    )
    if coef0 is not None:
        accepted_start = coef0.copy()
    if eta0 is not None:
        accepted_eta = eta0.copy()
    if mu0 is not None:
        accepted_mu = mu0.copy()

    if grad.ndim != 1:
        raise ValueError("Gradient must be a vector.")
    n = int(grad.size)
    if hess.shape != (n, n):
        raise ValueError("Hessian shape mismatch.")

    old_score = score + 1.0
    score_scale = _score_scale(score, old_score, scale_est=scale_est)
    uconv = np.abs(grad) > score_scale * conv_tol
    if dvkk.shape == grad.shape:
        uconv |= np.abs(dvkk) > score_scale * conv_tol * 0.1
    if not np.any(uconv):
        uconv = np.ones_like(uconv, dtype=bool)

    msg = "iteration limit reached"
    success = False
    step_failed = False
    nit = 0
    ii_last = 0

    for nit in range(1, int(max_iter) + 1):
        uconv_ind1 = uconv & (np.abs(grad) > (np.max(np.abs(grad)) * 1e-3))
        if not np.any(uconv_ind1):
            uconv_ind1 = uconv.copy()
        if not np.any(uconv):
            uconv[np.argmax(np.abs(grad))] = True

        hess1 = np.asarray(hess[np.ix_(uconv, uconv)], dtype=np.float64)
        grad1 = np.asarray(grad[uconv], dtype=np.float64)
        try:
            d, U = np.linalg.eigh(0.5 * (hess1 + hess1.T))
        except np.linalg.LinAlgError:
            step_failed = True
            msg = "eigendecomposition failed"
            break

        if d.size == 0:
            d = np.array([1.0], dtype=np.float64)
            U = np.ones((1, 1), dtype=np.float64)

        d0 = abs(float(d[0]))
        indef = bool(np.sum(-d > d0 * (np.finfo(np.float64).eps ** 0.5)) > 0)
        if indef and d.size == 1:
            indef = bool(d[0] < -(score_scale * (np.finfo(np.float64).eps ** 0.5)))
        neg = d < 0.0
        pdef = not bool(np.any(neg))
        d = d.copy()
        d[neg] = -d[neg]
        low_d = float(np.max(d)) * (np.finfo(np.float64).eps ** 0.7)
        too_low = d < low_d
        if np.any(too_low):
            pdef = False
            d[too_low] = low_d
        d_inv = 1.0 / d

        nstep = np.zeros_like(grad, dtype=np.float64)
        nstep[uconv] = -(U @ (d_inv * (U.T @ grad1)))
        sstep = -grad / max(float(np.max(np.abs(grad))), 1e-12)

        ms = float(np.max(np.abs(nstep)))
        if ms > float(max_nstep):
            nstep *= float(max_nstep) / ms

        sd_unused = True
        old_score = score
        x1 = _project_to_bounds(x + nstep, bounds)
        step1 = x1 - x
        trial_grad = None
        trial_hess = None
        trial_coef = None
        if not np.array_equal(x1, x):
            (
                score1,
                trial_grad,
                trial_hess,
                trial_dvkk,
                trial_coef,
                trial_eta,
                trial_mu,
                trial_scale,
            ) = _eval_at(
                x1,
                start_coef=accepted_start,
                start_eta=accepted_eta,
                start_mu=accepted_mu,
                need_grad=bool(pdef),
                need_hess=bool(pdef),
                commit_start=False,
            )
        else:
            score1 = np.inf
        pred_change = float(grad @ step1 + 0.5 * (step1 @ hess @ step1))
        score_change = float(score1 - score)
        denom = max(abs(pred_change), abs(score_change)) + score_scale * conv_tol
        qerror = abs(pred_change - score_change) / max(denom, 1e-12)
        ii = 0
        accepted = False
        trial_step_inf = float(np.max(np.abs(step1))) if step1.size else 0.0

        if (
            np.isfinite(score1)
            and score_change < 0.0
            and pdef
            and qerror < float(qerror_thresh)
        ):
            x = x1
            score = float(score1)
            grad = np.asarray(trial_grad, dtype=np.float64)
            hess = np.asarray(trial_hess, dtype=np.float64)
            dvkk = np.asarray(trial_dvkk, dtype=np.float64)
            if trial_coef is not None:
                accepted_start = np.asarray(trial_coef, dtype=np.float64).copy()
                model._pirls_coef_start_ = accepted_start.copy()
            if trial_eta is not None:
                accepted_eta = np.asarray(trial_eta, dtype=np.float64).copy()
                model._pirls_eta_start_ = accepted_eta.copy()
            if trial_mu is not None:
                accepted_mu = np.asarray(trial_mu, dtype=np.float64).copy()
                model._pirls_mu_start_ = accepted_mu.copy()
            accepted = True
        elif (
            (not np.isfinite(score1))
            or (score1 >= score)
            or (qerror >= float(qerror_thresh))
        ):
            step = nstep.copy()
            score2 = np.nan
            x2 = None
            while (
                (not np.isfinite(score1))
                or (score1 >= score)
                or (qerror >= float(qerror_thresh))
            ) and ii < int(max_half):
                if ii == 3 and nit < 10:
                    s_len = min(float(np.linalg.norm(step)), float(max_sstep))
                    s_den = max(float(np.linalg.norm(sstep)), 1e-12)
                    step = sstep * (s_len / s_den)
                    sd_unused = False
                else:
                    step *= 0.5

                x1 = _project_to_bounds(x + step, bounds)
                step1 = x1 - x
                trial_step_inf = max(
                    trial_step_inf,
                    float(np.max(np.abs(step1))) if step1.size else 0.0,
                )
                score1 = (
                    _eval_at(
                        x1,
                        start_coef=accepted_start,
                        start_eta=accepted_eta,
                        start_mu=accepted_mu,
                        need_grad=False,
                        need_hess=False,
                    )[0]
                    if not np.array_equal(x1, x)
                    else np.inf
                )
                pred_change = float(grad @ step1 + 0.5 * (step1 @ hess @ step1))
                score_change = float(score1 - score)
                if ii > min(4, int(max_half / 2)):
                    qerror = float(qerror_thresh) / 2.0
                else:
                    denom = (
                        max(abs(pred_change), abs(score_change))
                        + score_scale * conv_tol
                    )
                    qerror = abs(pred_change - score_change) / max(denom, 1e-12)

                if (
                    np.isfinite(score1)
                    and score_change < 0.0
                    and qerror < float(qerror_thresh)
                ):
                    if pdef or (not sd_unused):
                        x = x1
                        (
                            score,
                            grad,
                            hess,
                            dvkk,
                            coef_acc,
                            eta_acc,
                            mu_acc,
                            scale_est,
                        ) = _eval_at(
                            x,
                            start_coef=accepted_start,
                            start_eta=accepted_eta,
                            start_mu=accepted_mu,
                            need_grad=True,
                            need_hess=True,
                            commit_start=True,
                        )
                        if coef_acc is not None:
                            accepted_start = coef_acc.copy()
                        if eta_acc is not None:
                            accepted_eta = eta_acc.copy()
                        if mu_acc is not None:
                            accepted_mu = mu_acc.copy()
                        accepted = True
                    else:
                        x2 = x1.copy()
                        score2 = score1
                    score1 = score - abs(score) - 1.0
                if (
                    (not np.isfinite(score1))
                    or (score1 >= score)
                    or (qerror >= float(qerror_thresh))
                ):
                    ii += 1

            if (not pdef) and sd_unused and ii < int(max_half) and np.isfinite(score2):
                score1 = float(score2)
                x1 = x2.copy() if x2 is not None else x1

        if (not pdef) and sd_unused:
            step = sstep * 2.0
            kk = 0
            score2 = np.nan
            x2 = None
            while True:
                step *= 0.5
                kk += 1
                x3 = _project_to_bounds(x + step, bounds)
                step3 = x3 - x
                trial_step_inf = max(
                    trial_step_inf,
                    float(np.max(np.abs(step3))) if step3.size else 0.0,
                )
                score3 = (
                    _eval_at(
                        x3,
                        start_coef=accepted_start,
                        start_eta=accepted_eta,
                        start_mu=accepted_mu,
                        need_grad=False,
                        need_hess=False,
                    )[0]
                    if not np.array_equal(x3, x)
                    else np.inf
                )
                pred_change = float(grad @ step3 + 0.5 * (step3 @ hess @ step3))
                score_change = float(score3 - score)
                denom = (
                    max(abs(pred_change), abs(score_change)) + score_scale * conv_tol
                )
                qerror3 = abs(pred_change - score_change) / max(denom, 1e-12)
                if (not np.isfinite(score2)) or (
                    np.isfinite(score3)
                    and score3 <= score2
                    and qerror3 < float(qerror_thresh)
                ):
                    score2 = score3
                    x2 = x3.copy()

                if (
                    np.isfinite(score2)
                    and np.isfinite(score3)
                    and score2 < score
                    and score3 > score2
                ) or kk == 40:
                    break

            if np.isfinite(score2) and score2 < score1:
                score1 = score2
                x1 = x2.copy() if x2 is not None else x1

            if score1 < score and np.isfinite(score1):
                x = x1
                score, grad, hess, dvkk, coef_acc, eta_acc, mu_acc, scale_est = (
                    _eval_at(
                        x,
                        start_coef=accepted_start,
                        start_eta=accepted_eta,
                        start_mu=accepted_mu,
                        need_grad=True,
                        need_hess=True,
                        commit_start=True,
                    )
                )
                if coef_acc is not None:
                    accepted_start = coef_acc.copy()
                if eta_acc is not None:
                    accepted_eta = eta_acc.copy()
                if mu_acc is not None:
                    accepted_mu = mu_acc.copy()
                accepted = True

        if not accepted:
            if ii >= int(max_half) and trial_step_inf <= float(step_tol):
                success = True
                msg = "step tolerance satisfied"
                ii_last = ii
                break
            step_failed = True
            msg = "step failed"
            ii_last = ii
            break

        score_scale = _score_scale(score, old_score, scale_est=scale_est)
        grad2 = np.diag(hess)
        if grad2.shape != grad.shape or np.any(~np.isfinite(grad2)):
            grad2 = np.asarray(grad2, dtype=np.float64).reshape(-1)
        uconv = (np.abs(grad) > score_scale * conv_tol * 0.1) | (
            np.abs(grad2) > score_scale * conv_tol * 0.1
        )
        converged = not bool(indef)
        if np.any(np.abs(grad) > score_scale * conv_tol * 5.0):
            converged = False
        if abs(old_score - score) > score_scale * conv_tol:
            if converged:
                uconv = uconv | True
            converged = False
        if ii == int(max_half):
            converged = True
        ii_last = ii
        if converged:
            success = True
            msg = "full convergence"
            break

    if step_failed and not success:
        status = 1
    else:
        status = 0 if success else 1

    if (not success) and ii_last == int(max_half):
        msg = "step failed"
    elif (not success) and nit >= int(max_iter) and (not step_failed):
        msg = "iteration limit reached"

    return OptimizeResult(
        x=np.asarray(x, dtype=np.float64),
        fun=float(score),
        jac=np.asarray(grad, dtype=np.float64),
        hess=np.asarray(hess, dtype=np.float64),
        success=bool(success),
        status=status,
        message=msg,
        nit=int(nit),
        nfev=objective.n_fun,
        njev=objective.n_jac,
        nhev=objective.n_hess,
    )
