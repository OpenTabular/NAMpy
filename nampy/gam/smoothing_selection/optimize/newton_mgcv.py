"""Canonical `mgcv/R/gam.fit3.r::newton()` mirror for smoothing-parameter outer Newton."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import OptimizeResult

from ..._model_state import _fit_scale
from .basics import _project_to_bounds


def _mgcv_score_scale(
    score_type: str,
    score_val,
    old_score_val,
    *,
    model=None,
    scale_est=None,
):
    score_val = float(score_val)
    old_score_val = float(old_score_val)
    if str(score_type).upper() in {"REML", "P-REML", "ML", "P-ML"}:
        # Mirror `mgcv/R/gam.fit3.r::newton()`: for REML/ML-like scores use
        # `abs(log(scale.est)) + abs(score)` throughout.
        if scale_est is None:
            scale_obj = _fit_scale(model)
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


def _optimize_outer_newton_mgcv(
    objective,
    x0,
    bounds,
    eval_at,
    *,
    score_type="reml",
    conv_tol=1e-6,
    max_nstep=5.0,
    max_sstep=2.0,
    max_half=30,
    qerror_thresh=0.8,
    max_iter=200,
    edge_correct=False,
    step_tol=1e-7,
):
    """Python translation of `mgcv/R/gam.fit3.r::newton`.

    Arguments are kept intentionally close to the historical mgcv signature while
    preserving existing Python model/objective plumbing.
    """

    # Mirror: `lsp` in R is constrained by bounds here.
    x = _project_to_bounds(np.asarray(x0, dtype=np.float64), bounds)
    initial_lsp = x.copy()
    accepted_start = None
    accepted_eta = None
    accepted_mu = None

    (
        score,
        grad,
        hess,
        dvkk,
        coef0,
        eta0,
        mu0,
        scale_est,
    ) = eval_at(
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

    old_score = score
    score_scale = _mgcv_score_scale(
        score_type,
        score,
        old_score,
        model=getattr(objective, "model", None),
        scale_est=scale_est,
    )

    uconv = np.abs(grad) > score_scale * conv_tol
    if not np.any(uconv):
        uconv = np.ones_like(uconv, dtype=bool)

    score_hist = np.full(int(max_iter), np.nan, dtype=np.float64)
    accepted_x_hist = []
    iter_trace = []

    msg = "iteration limit reached"
    success = False
    step_failed = False
    nit = 0
    ii_last = 0

    def _curvature_diag(dvkk_val, hess_val, shape):
        del dvkk_val
        hess_arr = np.asarray(hess_val, dtype=np.float64)
        if hess_arr.shape == (shape[0], shape[0]):
            out = np.diag(hess_arr).astype(np.float64, copy=True)
            if out.shape == shape and np.all(np.isfinite(out)):
                return out
        return np.full(shape, np.nan, dtype=np.float64)

    def _record_iter(x_prev, x_next):
        if not hasattr(objective, "record_iter"):
            return
        x_prev = np.asarray(x_prev, dtype=np.float64).ravel()
        x_next = np.asarray(x_next, dtype=np.float64).ravel()
        if x_prev.shape != x_next.shape:
            return
        try:
            step_norm = float(np.linalg.norm(x_next - x_prev))
            objective.record_iter(x_next, step_norm)
        except Exception:
            return

    for nit in range(1, int(max_iter) + 1):
        uconv_ind1 = uconv & (np.abs(grad) > (np.max(np.abs(grad)) * 1e-3))
        if not np.any(uconv_ind1):
            uconv_ind1 = uconv.copy()
        if not np.any(uconv):
            uconv[np.argmax(np.abs(grad))] = True

        hess1 = np.asarray(hess[np.ix_(uconv, uconv)], dtype=np.float64)
        grad1 = np.asarray(grad[uconv], dtype=np.float64)
        try:
            d_full, U_full = scipy_eigh(
                hess1,
                check_finite=False,
                driver="evr",
            )
            d = d_full[::-1].copy()
            U = U_full[:, ::-1].copy()
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
            indef = bool(
                float(d[0]) < -(score_scale * (np.finfo(np.float64).eps ** 0.5))
            )
        neg = d < 0.0
        pdef = not bool(np.any(neg))
        d = d.copy()
        d[neg] = -d[neg]
        low_d = float(np.max(d)) * (np.finfo(np.float64).eps ** 0.7)
        too_low = d < low_d
        if np.any(too_low):
            pdef = False
            d[too_low] = low_d
        d_inv = np.zeros_like(d)
        nonzero = d != 0.0
        d_inv[nonzero] = 1.0 / d[nonzero]

        nstep = np.zeros_like(grad, dtype=np.float64)
        nstep[uconv] = -(U @ (d_inv * (U.T @ grad1)))
        sstep = grad / max(float(np.max(np.abs(grad))), 1e-12)

        ms = float(np.max(np.abs(nstep))) if nstep.size else 0.0
        if ms > float(max_nstep):
            nstep *= float(max_nstep) / ms

        sd_unused = True
        old_score = score
        x1 = _project_to_bounds(x + nstep, bounds)
        step1 = x1 - x
        trial_grad = None
        trial_hess = None
        if not np.array_equal(x1, x):
            (
                score1,
                trial_grad,
                trial_hess,
                trial_dvkk,
                trial_coef,
                trial_eta,
                trial_mu,
                trial_scale_est,
            ) = eval_at(
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
            trial_dvkk = np.full_like(x1, np.nan)
            trial_coef = None
            trial_eta = None
            trial_mu = None
            trial_scale_est = scale_est

        pred_change = float(grad @ step1 + 0.5 * (step1 @ hess @ step1))
        score_change = float(score1 - score)
        denom = max(abs(pred_change), abs(score_change)) + score_scale * conv_tol
        qerror = abs(pred_change - score_change) / max(denom, 1e-12)
        ii = 0
        accepted = False
        trial_step_inf = float(np.max(np.abs(step1))) if step1.size else 0.0
        used_sd_step = False

        if (
            np.isfinite(score1)
            and score_change < 0.0
            and pdef
            and qerror < float(qerror_thresh)
        ):
            _record_iter(x, x1)
            old_score = float(score)
            x = x1
            score = float(score1)
            grad = np.asarray(trial_grad, dtype=np.float64)
            hess = np.asarray(trial_hess, dtype=np.float64)
            dvkk = np.asarray(trial_dvkk, dtype=np.float64)
            scale_est = trial_scale_est
            if trial_coef is not None:
                accepted_start = np.asarray(trial_coef, dtype=np.float64).copy()
            if trial_eta is not None:
                accepted_eta = np.asarray(trial_eta, dtype=np.float64).copy()
            if trial_mu is not None:
                accepted_mu = np.asarray(trial_mu, dtype=np.float64).copy()
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
                    used_sd_step = True
                else:
                    step *= 0.5

                x1 = _project_to_bounds(x + step, bounds)
                step1 = x1 - x
                trial_step_inf = max(
                    trial_step_inf,
                    float(np.max(np.abs(step1))) if step1.size else 0.0,
                )
                score1 = (
                    eval_at(
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
                        prev_score = float(score)
                        (
                            score,
                            grad,
                            hess,
                            dvkk,
                            coef_acc,
                            eta_acc,
                            mu_acc,
                            scale_est,
                        ) = eval_at(
                            x1,
                            start_coef=accepted_start,
                            start_eta=accepted_eta,
                            start_mu=accepted_mu,
                            need_grad=True,
                            need_hess=True,
                            commit_start=True,
                        )
                        if coef_acc is not None:
                            accepted_start = np.asarray(
                                coef_acc, dtype=np.float64
                            ).copy()
                        if eta_acc is not None:
                            accepted_eta = np.asarray(eta_acc, dtype=np.float64).copy()
                        if mu_acc is not None:
                            accepted_mu = np.asarray(mu_acc, dtype=np.float64).copy()
                        accepted = True
                        _record_iter(x, x1)
                        old_score = prev_score
                        x = x1
                    else:
                        x2 = x1.copy()
                        score2 = float(score1)
                    score1 = score - abs(score) - 1.0
                if (
                    (not np.isfinite(score1))
                    or (score1 >= score)
                    or (qerror >= float(qerror_thresh))
                ):
                    ii += 1
            if (not pdef) and sd_unused and ii < int(max_half) and np.isfinite(score2):
                x1 = x2.copy() if x2 is not None else x1
                score1 = float(score2)

        if (not pdef) and sd_unused:
            step = sstep * 2.0
            kk = 0
            score2 = np.nan
            x2 = None
            ok = True
            while ok:
                step *= 0.5
                kk += 1
                x3 = _project_to_bounds(x + step, bounds)
                step3 = x3 - x
                trial_step_inf = max(
                    trial_step_inf,
                    float(np.max(np.abs(step3))) if step3.size else 0.0,
                )
                score3 = (
                    eval_at(
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
                qerror3 = abs(pred_change - score_change) / (
                    max(abs(pred_change), abs(score_change))
                    + score_scale * conv_tol
                    + 1e-12
                )
                if (not np.isfinite(score2)) or (
                    np.isfinite(score3)
                    and score3 <= score2
                    and qerror3 < float(qerror_thresh)
                ):
                    score2 = float(score3)
                    x2 = x3.copy()

                if (
                    np.isfinite(score2)
                    and np.isfinite(score3)
                    and score2 < score
                    and score3 > score2
                ) or kk == 40:
                    ok = False

            if np.isfinite(score2) and score2 < score1:
                score1 = score2
                x1 = x2.copy() if x2 is not None else x1
                used_sd_step = True

            if score1 < score and np.isfinite(score1):
                prev_score = float(score)
                (
                    score,
                    grad,
                    hess,
                    dvkk,
                    coef_acc,
                    eta_acc,
                    mu_acc,
                    scale_est,
                ) = eval_at(
                    x1,
                    start_coef=accepted_start,
                    start_eta=accepted_eta,
                    start_mu=accepted_mu,
                    need_grad=True,
                    need_hess=True,
                    commit_start=True,
                )
                if coef_acc is not None:
                    accepted_start = np.asarray(coef_acc, dtype=np.float64).copy()
                if eta_acc is not None:
                    accepted_eta = np.asarray(eta_acc, dtype=np.float64).copy()
                if mu_acc is not None:
                    accepted_mu = np.asarray(mu_acc, dtype=np.float64).copy()
                accepted = True
                _record_iter(x, x1)
                old_score = prev_score
                x = x1

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

        # Mirror mgcv: record current score each outer iteration.
        if nit - 1 < len(score_hist):
            score_hist[nit - 1] = float(score)
            accepted_x_hist.append(np.asarray(x, dtype=np.float64).copy())

        score_scale = _mgcv_score_scale(
            score_type,
            score,
            old_score,
            model=getattr(objective, "model", None),
            scale_est=scale_est,
        )
        grad2 = _curvature_diag(dvkk, hess, grad.shape)
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
        iter_trace.append(
            {
                "iter": int(nit),
                "log_sp": np.asarray(x, dtype=np.float64).copy(),
                "criterion": float(score),
                "gradient": np.asarray(grad, dtype=np.float64).copy(),
                "hessian": np.asarray(hess, dtype=np.float64).copy(),
                "accepted_step_norm": (
                    0.0
                    if len(accepted_x_hist) <= 1
                    else float(
                        np.linalg.norm(
                            np.asarray(accepted_x_hist[-1], dtype=np.float64)
                            - np.asarray(accepted_x_hist[-2], dtype=np.float64)
                        )
                    )
                ),
                "rank_info": {
                    "source": "outer_newton_mgcv",
                    "indefinite_hessian": bool(indef),
                    "positive_definite": bool(pdef),
                    "step_halving_count": int(ii),
                    "used_steepest_descent": bool(used_sd_step),
                    "converged_here": bool(converged),
                },
            }
        )
        if converged:
            success = True
            msg = "full convergence"
            break

        if int(nit) >= max_iter:
            break

    status = 1 if (step_failed and not success) else 0 if success else 1

    if ii_last == int(max_half):
        msg = "step failed"
    elif (not success) and nit >= max_iter and (not step_failed):
        msg = "iteration limit reached"

    edge_corrected = False
    hess1 = None
    lsp1 = None
    db_drho1 = None
    dw_drho1 = None
    rp1 = None
    reml_mode = str(score_type).upper() in {"REML", "P-REML", "ML", "P-ML"}
    grad2 = _curvature_diag(dvkk, hess, grad.shape)
    if not np.all(np.isfinite(grad2)):
        grad2 = np.zeros_like(grad, dtype=np.float64)
    if bool(edge_correct) and reml_mode and grad.size > 0:
        if isinstance(edge_correct, (bool, np.bool_)) and bool(edge_correct):
            alpha = 0.02
        else:
            alpha = abs(float(edge_correct))
            if not np.isfinite(alpha):
                alpha = 0.02
            if alpha <= 0.0:
                alpha = 0.02
        flat = np.flatnonzero(np.abs(grad2) < np.abs(grad) * 100.0)
        lsp1 = np.asarray(x, dtype=np.float64).copy()
        lsp_step = np.where(initial_lsp > lsp1, 1.0, -1.0)
        if flat.size:
            target = score + alpha
            for i in flat:
                step_i = float(lsp_step[i])
                score1 = np.inf
                for _ in range(1, 20000):
                    lsp2 = _project_to_bounds(lsp1 + 0.0, bounds)
                    lsp2[i] = float(lsp2[i] + step_i)
                    lsp2 = _project_to_bounds(lsp2, bounds)
                    if np.array_equal(lsp2, lsp1):
                        break
                    score1 = eval_at(
                        lsp2,
                        start_coef=accepted_start,
                        start_eta=accepted_eta,
                        start_mu=accepted_mu,
                        need_grad=False,
                        need_hess=False,
                        commit_start=False,
                    )[0]
                    lsp1 = lsp2
                    if not np.isfinite(score1):
                        break
                    if score1 < target:
                        continue
                    break
        (
            score1,
            grad1,
            hess1,
            dvkk1,
            coef1,
            eta1,
            mu1,
            _,
        ) = eval_at(
            lsp1,
            start_coef=accepted_start,
            start_eta=accepted_eta,
            start_mu=accepted_mu,
            need_grad=True,
            need_hess=True,
            commit_start=False,
        )
        if np.isfinite(score1):
            edge_corrected = True
            model = getattr(objective, "model", None)
            if model is not None and isinstance(
                getattr(model, "_pirls_reml_derivative_kernel_state_", None), dict
            ):
                drv = model._pirls_reml_derivative_kernel_state_
                # `db.drho`/`dw.drho`/`rp` are not yet carried through Python's
                # PIRLS exact path in the same raw form as mgcv.
                db_drho1 = drv.get("dbeta")
                dw_drho1 = drv.get("dW_obs")
                rp1 = drv.get("rp")
            elif model is not None and isinstance(
                getattr(model, "_general_family_outer_derivative_info", None), dict
            ):
                drv = model._general_family_outer_derivative_info
                db_drho1 = drv.get("db_drho")
        else:
            hess1 = None
    elif bool(edge_correct) and reml_mode and grad.size == 0:
        lsp1 = np.asarray(x, dtype=np.float64).copy()
        hess1 = np.asarray(hess, dtype=np.float64).copy()
        edge_corrected = True

    result = OptimizeResult(
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
    # Mirror `mgcv/R/gam.fit3.r::newton()`: expose in-loop `score.hist`,
    # not post-hoc objective recomputes at accepted iterates.
    result.mgcv_score_hist = [v for v in score_hist.tolist() if np.isfinite(v)]
    result.accepted_x_hist = accepted_x_hist
    result.optim_trace = iter_trace
    result.mgcv_qerror_thresh = float(qerror_thresh)
    result.mgcv_edge_correct = bool(edge_correct)
    result.mgcv_edge_correct_applied = bool(edge_corrected)
    result.hess1 = None if hess1 is None else np.asarray(hess1, dtype=np.float64)
    result.db_drho1 = (
        None if db_drho1 is None else np.asarray(db_drho1, dtype=np.float64)
    )
    result.dw_drho1 = (
        None if dw_drho1 is None else np.asarray(dw_drho1, dtype=np.float64)
    )
    result.rp = rp1
    result.lsp1 = None if lsp1 is None else np.asarray(lsp1, dtype=np.float64)
    result.outer_info = {
        "conv": str(msg),
        "iter": int(nit),
        "score_hist": result.mgcv_score_hist,
        "grad": np.asarray(grad, dtype=np.float64),
        "hess": np.asarray(hess, dtype=np.float64),
        "convergence": int(status),
        "message": str(msg),
        "counts": np.asarray(
            [int(objective.n_fun), int(objective.n_jac)],
            dtype=np.int64,
        ),
    }
    if bool(edge_correct):
        result.outer_info.update(
            {
                "hess1": (
                    None if hess1 is None else np.asarray(hess1, dtype=np.float64)
                ),
                "db_drho1": (
                    None if db_drho1 is None else np.asarray(db_drho1, dtype=np.float64)
                ),
                "dw_drho1": (
                    None if dw_drho1 is None else np.asarray(dw_drho1, dtype=np.float64)
                ),
                "rp": rp1,
                "lsp1": None if lsp1 is None else np.asarray(lsp1, dtype=np.float64),
            }
        )
    return result
