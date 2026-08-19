"""Canonical `mgcv/R/gam.fit3.r::bfgs()` mirror for outer smoothing selection."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import OptimizeResult

from ....model_state import _fit_scale, _fit_workspace
from .basics import _project_to_bounds


def _copy_state_vector(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def _extract_scale_estimate(model, objective=None):
    gamma_state = _fit_workspace(model).get("pirls_reml_gamma_state", None)
    if isinstance(gamma_state, dict):
        scale_est = gamma_state.get("scale_est", None)
        if scale_est is not None and np.isfinite(scale_est) and float(scale_est) > 0.0:
            return float(scale_est)

    if bool(getattr(objective, "uses_joint_log_scale", False)):
        scale_obj = getattr(model, "_gaussian_reml_last_scale_est_", None)
        if scale_obj is not None and np.isfinite(scale_obj) and float(scale_obj) > 0.0:
            return float(scale_obj)

    scale_obj = _fit_scale(model)
    if scale_obj is not None and np.isfinite(scale_obj) and float(scale_obj) > 0.0:
        return float(scale_obj)

    known_scale = getattr(getattr(model, "family", None), "known_scale", None)
    if (
        known_scale is not None
        and np.isfinite(known_scale)
        and float(known_scale) > 0.0
    ):
        return float(known_scale)
    return None


def _extract_dvkk_diag(model, x_eval, hess):
    kernel_state = _fit_workspace(model).get("pirls_reml_derivative_kernel_state", None)
    if isinstance(kernel_state, dict):
        dvkk = kernel_state.get("dVkk", None)
        if dvkk is not None:
            dvkk_arr = np.asarray(dvkk, dtype=np.float64)
            if dvkk_arr.shape == np.asarray(x_eval, dtype=np.float64).ravel().shape:
                return dvkk_arr.ravel().copy()
            if dvkk_arr.ndim == 1 and 0 < dvkk_arr.size <= np.asarray(
                x_eval, dtype=np.float64
            ).ravel().size:
                return dvkk_arr.copy()
            if dvkk_arr.ndim == 2 and dvkk_arr.shape[0] == dvkk_arr.shape[1]:
                return np.asarray(np.diag(dvkk_arr), dtype=np.float64).copy()
    if hess is not None:
        hess = np.asarray(hess, dtype=np.float64)
        if hess.ndim == 2 and hess.shape[0] == hess.shape[1]:
            diag = np.diag(hess).astype(np.float64, copy=True)
            if diag.shape == np.asarray(x_eval, dtype=np.float64).ravel().shape:
                return diag
    return np.full(np.asarray(x_eval, dtype=np.float64).ravel().shape, np.nan)


def _eval_objective_at(
    objective,
    x_eval,
    *,
    start_coef=None,
    start_eta=None,
    start_mu=None,
    need_grad=False,
    need_hess=False,
    need_score=True,
    commit_start=False,
):
    model = getattr(objective, "model", None)
    x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
    prev_coef_start = None
    prev_eta_start = None
    prev_mu_start = None
    if model is not None:
        prev_coef_start = _copy_state_vector(_fit_workspace(model).get("pirls_coef_start", None))
        prev_eta_start = _copy_state_vector(_fit_workspace(model).get("pirls_eta_start", None))
        prev_mu_start = _copy_state_vector(_fit_workspace(model).get("pirls_mu_start", None))
        if start_coef is None:
            start_coef = _fit_workspace(model).get("pirls_coef_start", None)
        if start_mu is None:
            start_mu = _fit_workspace(model).get("pirls_mu_start", None)
        _fit_workspace(model).pirls_eval_start = _copy_state_vector(start_coef)
        # mgcv::bfgs carries coefficient and working-response starts, but not
        # etastart, between outer evaluations.
        _fit_workspace(model).pirls_eval_eta_start = None
        _fit_workspace(model).pirls_eval_mu_start = _copy_state_vector(start_mu)
        _fit_workspace(model).pirls_coef_start = _copy_state_vector(start_coef)
        _fit_workspace(model).pirls_eta_start = None
        _fit_workspace(model).pirls_mu_start = _copy_state_vector(start_mu)
        _fit_workspace(model).pirls_lock_start = True

    objective._last_x = None
    objective._last_fun = None
    objective._last_grad = None
    objective._last_hess = None

    try:
        score = np.nan
        grad = None
        if bool(need_score):
            score = float(objective.fun(x_eval))
        grad = (
            np.asarray(objective.jac(x_eval), dtype=np.float64)
            if need_grad
            else None
        )
        hess = (
            np.asarray(objective.hess(x_eval), dtype=np.float64) if need_hess else None
        )
        if (
            getattr(objective, "_last_fun", None) is not None
            and hasattr(objective, "_same_x")
            and bool(objective._same_x(x_eval))
        ):
            score = float(objective._last_fun)
        hess_for_dvkk = hess
        if need_grad and model is not None and hess_for_dvkk is None:
            dvkk_now = _extract_dvkk_diag(model, x_eval, None)
            if not np.all(np.isfinite(dvkk_now)):
                hess_method = getattr(objective, "hess", None)
                if hess_method is not None:
                    hess_for_dvkk = np.asarray(hess_method(x_eval), dtype=np.float64)

        coef_eval = (
            _fit_workspace(model).get("pirls_last_coef", None) if model is not None else None
        )
        eta_eval = (
            _fit_workspace(model).get("pirls_last_eta", None) if model is not None else None
        )
        mu_eval = _fit_workspace(model).get("pirls_last_mu", None) if model is not None else None
        if coef_eval is not None:
            coef_eval = np.asarray(coef_eval, dtype=np.float64).copy()
            if commit_start:
                _fit_workspace(model).pirls_coef_start = coef_eval.copy()
        if eta_eval is not None:
            eta_eval = np.asarray(eta_eval, dtype=np.float64).copy()
        if mu_eval is not None:
            mu_eval = np.asarray(mu_eval, dtype=np.float64).copy()
            if commit_start:
                _fit_workspace(model).pirls_mu_start = mu_eval.copy()

        scale_est = None if model is None else _extract_scale_estimate(model, objective)
        dvkk = (
            np.full(x_eval.shape, np.nan, dtype=np.float64)
            if model is None
            else _extract_dvkk_diag(model, x_eval, hess_for_dvkk)
        )
    finally:
        if model is not None:
            _fit_workspace(model).pirls_eval_start = None
            _fit_workspace(model).pirls_eval_eta_start = None
            _fit_workspace(model).pirls_eval_mu_start = None
            _fit_workspace(model).pirls_lock_start = False
            if not commit_start:
                _fit_workspace(model).pirls_coef_start = prev_coef_start
                _fit_workspace(model).pirls_eta_start = prev_eta_start
                _fit_workspace(model).pirls_mu_start = prev_mu_start
            else:
                _fit_workspace(model).pirls_eta_start = None

    return score, grad, hess, dvkk, coef_eval, eta_eval, mu_eval, scale_est


def _bfgs_score_scale(score_type: str, score_val, *, scale_est=None, model=None):
    score_val = float(score_val)
    if str(score_type).upper() in {"REML", "P-REML", "ML", "P-ML"}:
        return float(1.0 + abs(score_val))
    if scale_est is None and model is not None:
        scale_est = _extract_scale_estimate(model)
    scale_abs = 0.0 if scale_est is None else abs(float(scale_est))
    score_scale = scale_abs + abs(score_val)
    return float(score_scale if score_scale > 0.0 else 1.0)


def _finite_difference_initial_inverse_hessian(
    objective,
    x0,
    grad0,
    *,
    start_coef,
    start_eta,
    start_mu,
    feps=1e-4,
):
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    grad0 = np.asarray(grad0, dtype=np.float64).ravel()
    n = int(x0.size)
    B = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        x1 = x0.copy()
        x1[i] += float(feps)
        _, grad1, _, _, _, _, _, _ = _eval_objective_at(
            objective,
            x1,
            start_coef=start_coef,
            start_eta=start_eta,
            start_mu=start_mu,
            need_grad=True,
            need_score=False,
            commit_start=False,
        )
        B[i, :] = (np.asarray(grad1, dtype=np.float64) - grad0) / float(feps)

    B = 0.5 * (B + B.T)
    evals, evecs = scipy_eigh(B, check_finite=False)
    evals = np.abs(evals)
    thresh = float(np.max(evals)) * 1e-4 if evals.size else 1.0
    if thresh <= 0.0:
        thresh = 1e-4
    evals = np.where(evals < thresh, thresh, evals)
    return np.asarray(evecs @ ((1.0 / evals)[:, None] * evecs.T), dtype=np.float64)


def _invert_inverse_hessian(B_inv):
    B_inv = 0.5 * (
        np.asarray(B_inv, dtype=np.float64) + np.asarray(B_inv, dtype=np.float64).T
    )
    evals, evecs = scipy_eigh(B_inv, check_finite=False)
    if evals.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    keep = evals > float(np.max(evals)) * (np.finfo(np.float64).eps ** 0.9)
    vals = np.zeros_like(evals)
    vals[keep] = 1.0 / evals[keep]
    return np.asarray(evecs @ (vals[:, None] * evecs.T), dtype=np.float64)


def _optimize_outer_bfgs_strict(
    objective,
    x0,
    bounds,
    *,
    score_type="reml",
    conv_tol=1e-6,
    # `mgcv::gam()` passes `gam.control()$newton$maxNstep`, whose public
    # default is 5, overriding the standalone `mgcv:::bfgs()` default of 3.
    max_nstep=5.0,
    max_sstep=2.0,
    max_step=200,
):
    """Python translation of `mgcv/R/gam.fit3.r::bfgs()`."""

    model = getattr(objective, "model", None)
    prev_irls_tol = None
    if model is not None:
        prev_irls_tol = float(getattr(model, "irls_tol", 1e-7))
        if prev_irls_tol > float(conv_tol) / 100.0:
            model.irls_tol = float(conv_tol) / 100.0

    try:
        x0 = _project_to_bounds(np.asarray(x0, dtype=np.float64), bounds)
        n = int(x0.size)
        if n == 0:
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

        (
            score0,
            grad0,
            _,
            dvkk0,
            coef0,
            eta0,
            mu0,
            scale0,
        ) = _eval_objective_at(
            objective,
            x0,
            need_grad=True,
            commit_start=True,
        )
        initial = {
            "alpha": 0.0,
            "score": float(score0),
            "grad": np.asarray(grad0, dtype=np.float64),
            "dVkk": np.asarray(dvkk0, dtype=np.float64),
            "start": _copy_state_vector(coef0),
            "eta": _copy_state_vector(eta0),
            "mustart": _copy_state_vector(mu0),
            "scale_est": scale0,
        }
        B = _finite_difference_initial_inverse_hessian(
            objective,
            x0,
            np.asarray(grad0, dtype=np.float64),
            start_coef=_copy_state_vector(coef0),
            start_eta=_copy_state_vector(eta0),
            start_mu=_copy_state_vector(mu0),
        )

        c1 = 1e-4
        c2 = 0.9
        score_scale = _bfgs_score_scale(
            score_type,
            initial["score"],
            scale_est=initial["scale_est"],
            model=getattr(objective, "model", None),
        )
        initial_lsp = x0.copy()
        lsp = x0.copy()
        ilsp = x0.copy()
        score_hist = np.full(int(max_step) + 1, np.nan, dtype=np.float64)
        score_hist[0] = float(initial["score"])
        iter_trace = []
        rolled_back = False
        uconv = np.ones(n, dtype=bool)
        trial = initial.copy()
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        uses_joint_log_theta = bool(getattr(objective, "uses_joint_log_theta", False))
        joint_log_theta_first = bool(
            getattr(objective, "joint_log_theta_first", False)
        )

        def _apply_dvkk_signal(mask, dvkk, factor):
            out = np.asarray(mask, dtype=bool).copy()
            dvkk_arr = np.asarray(dvkk, dtype=np.float64).ravel()
            if dvkk_arr.size == 0:
                return out
            if uses_joint_log_scale and dvkk_arr.size >= n:
                ind = np.arange(max(0, n - 1), dtype=np.int64)
                vals = dvkk_arr[: ind.size]
            elif (
                uses_joint_log_theta
                and joint_log_theta_first
                and dvkk_arr.size >= n
            ):
                ind = np.arange(1, n, dtype=np.int64)
                vals = dvkk_arr[1 : 1 + ind.size]
            else:
                ind = np.arange(min(dvkk_arr.size, n), dtype=np.int64)
                vals = dvkk_arr[: ind.size]
            if ind.size:
                out[ind] = out[ind] | (
                    np.abs(vals) > score_scale * conv_tol * float(factor)
                )
            return out

        def zoom(step, lo, hi):
            for _ in range(40):
                alpha = 0.5 * (float(lo["alpha"]) + float(hi["alpha"]))
                lsp_trial = _project_to_bounds(ilsp + step * alpha, bounds)
                (
                    score_val,
                    _,
                    _,
                    _,
                    coef_trial,
                    eta_trial,
                    mu_trial,
                    scale_trial,
                ) = _eval_objective_at(
                    objective,
                    lsp_trial,
                    start_coef=initial["start"],
                    start_eta=initial["eta"],
                    start_mu=initial["mustart"],
                    need_grad=False,
                    commit_start=False,
                )
                trial_local = {
                    "alpha": float(alpha),
                    "score": float(score_val),
                    "start": _copy_state_vector(coef_trial),
                    "eta": _copy_state_vector(eta_trial),
                    "mustart": _copy_state_vector(mu_trial),
                    "scale_est": scale_trial,
                }
                if trial_local["score"] > float(initial["score"]) + alpha * c1 * float(
                    initial["dscore"]
                ) or trial_local["score"] >= float(lo["score"]):
                    hi = trial_local
                else:
                    (
                        _,
                        grad_trial,
                        _,
                        dvkk_trial,
                        coef_grad,
                        eta_grad,
                        mu_grad,
                        scale_grad,
                    ) = _eval_objective_at(
                        objective,
                        lsp_trial,
                        start_coef=initial["start"],
                        start_eta=initial["eta"],
                        start_mu=initial["mustart"],
                        need_grad=True,
                        commit_start=False,
                    )
                    trial_local["grad"] = np.asarray(grad_trial, dtype=np.float64)
                    trial_local["dVkk"] = np.asarray(dvkk_trial, dtype=np.float64)
                    trial_local["dscore"] = float(step @ trial_local["grad"])
                    trial_local["start"] = _copy_state_vector(coef_grad)
                    trial_local["eta"] = _copy_state_vector(eta_grad)
                    trial_local["mustart"] = _copy_state_vector(mu_grad)
                    trial_local["scale_est"] = scale_grad
                    if abs(float(trial_local["dscore"])) <= -c2 * float(
                        initial["dscore"]
                    ):
                        return trial_local
                    if (
                        float(trial_local["dscore"])
                        * (float(hi["alpha"]) - float(lo["alpha"]))
                        >= 0.0
                    ):
                        hi = lo
                    lo = trial_local
            return None

        for i in range(1, int(max_step) + 1):
            step = np.zeros(n, dtype=np.float64)
            if np.any(uconv):
                step_u = -(
                    B[np.ix_(uconv, uconv)]
                    @ np.asarray(initial["grad"], dtype=np.float64)[uconv][:, None]
                )
                step[np.asarray(uconv, dtype=bool)] = np.asarray(
                    step_u[:, 0], dtype=np.float64
                )
            if float(step @ np.asarray(initial["grad"], dtype=np.float64)) >= 0.0:
                step = -np.diag(B) * np.asarray(initial["grad"], dtype=np.float64)
                step[~uconv] = 0.0

            ms = float(np.max(np.abs(step))) if step.size else 0.0
            if ms <= 0.0:
                trial = initial.copy()
                converged = True
            else:
                if ms > float(max_nstep):
                    alpha = float(max_nstep) / ms
                    alpha_max = alpha * 1.05
                else:
                    alpha = 1.0
                    alpha_max = min(2.0, float(max_nstep) / ms)
                initial["dscore"] = float(
                    step @ np.asarray(initial["grad"], dtype=np.float64)
                )
                prev = initial.copy()
                deriv = 1
                while True:
                    lsp = _project_to_bounds(ilsp + alpha * step, bounds)
                    (
                        score_val,
                        grad_val,
                        _,
                        dvkk_val,
                        coef_trial,
                        eta_trial,
                        mu_trial,
                        scale_trial,
                    ) = _eval_objective_at(
                        objective,
                        lsp,
                        start_coef=prev["start"],
                        start_eta=prev["eta"],
                        start_mu=prev["mustart"],
                        need_grad=bool(deriv > 0),
                        commit_start=False,
                    )
                    trial = {
                        "alpha": float(alpha),
                        "score": float(score_val),
                        "start": _copy_state_vector(coef_trial),
                        "eta": _copy_state_vector(eta_trial),
                        "mustart": _copy_state_vector(mu_trial),
                        "scale_est": scale_trial,
                    }
                    if deriv > 0 and grad_val is not None:
                        trial["grad"] = np.asarray(grad_val, dtype=np.float64)
                        trial["dVkk"] = np.asarray(dvkk_val, dtype=np.float64)
                        trial["dscore"] = float(trial["grad"] @ step)
                        deriv = 0

                    if (
                        trial["score"]
                        > float(initial["score"])
                        + c1 * float(trial["alpha"]) * float(initial["dscore"])
                    ) or (deriv == 0 and trial["score"] >= float(prev["score"])):
                        trial = zoom(step, prev, trial)
                        break

                    if "dscore" not in trial:
                        (
                            _,
                            grad_now,
                            _,
                            dvkk_now,
                            coef_now,
                            eta_now,
                            mu_now,
                            scale_now,
                        ) = _eval_objective_at(
                            objective,
                            lsp,
                            start_coef=trial["start"],
                            start_eta=trial["eta"],
                            start_mu=trial["mustart"],
                            need_grad=True,
                            commit_start=False,
                        )
                        trial["grad"] = np.asarray(grad_now, dtype=np.float64)
                        trial["dVkk"] = np.asarray(dvkk_now, dtype=np.float64)
                        trial["dscore"] = float(trial["grad"] @ step)
                        trial["start"] = _copy_state_vector(coef_now)
                        trial["eta"] = _copy_state_vector(eta_now)
                        trial["mustart"] = _copy_state_vector(mu_now)
                        trial["scale_est"] = scale_now

                    if abs(float(trial["dscore"])) <= -c2 * float(initial["dscore"]):
                        break
                    if float(trial["dscore"]) >= 0.0:
                        trial = zoom(step, trial, prev)
                        break
                    prev = trial
                    if float(trial["alpha"]) == float(alpha_max):
                        break
                    alpha = min(float(prev["alpha"]) * 1.3, float(alpha_max))

                if trial is None:
                    lsp = ilsp.copy()
                    (
                        score_curr,
                        grad_curr,
                        _,
                        dvkk_curr,
                        coef_curr,
                        eta_curr,
                        mu_curr,
                        scale_curr,
                    ) = _eval_objective_at(
                        objective,
                        ilsp,
                        start_coef=initial["start"],
                        start_eta=initial["eta"],
                        start_mu=initial["mustart"],
                        need_grad=True,
                        commit_start=False,
                    )
                    initial["score"] = float(score_curr)
                    initial["grad"] = np.asarray(grad_curr, dtype=np.float64)
                    initial["dVkk"] = np.asarray(dvkk_curr, dtype=np.float64)
                    initial["start"] = _copy_state_vector(coef_curr)
                    initial["eta"] = _copy_state_vector(eta_curr)
                    initial["mustart"] = _copy_state_vector(mu_curr)
                    initial["scale_est"] = scale_curr
                    ilsp = ilsp.copy()
                    if rolled_back:
                        break
                    uconv = np.abs(np.asarray(initial["grad"], dtype=np.float64)) > (
                        score_scale * conv_tol * 0.1
                    )
                    uconv = _apply_dvkk_signal(uconv, initial["dVkk"], 0.1)
                    if np.all(uconv):
                        break
                    trial = initial.copy()
                    converged = True
                else:
                    yg = np.asarray(trial["grad"], dtype=np.float64) - np.asarray(
                        initial["grad"], dtype=np.float64
                    )
                    step = step * float(trial["alpha"])
                    rho = float(yg @ step)
                    if rho > 0.0:
                        if i == 1:
                            B = B * float(trial["alpha"])
                        rho = 1.0 / rho
                        step_col = np.asarray(step[:, None], dtype=np.float64)
                        yg_col = np.asarray(yg[:, None], dtype=np.float64)
                        B = B - rho * (
                            step_col @ (yg_col.T @ np.asarray(B, dtype=np.float64))
                        )
                        B = (
                            B
                            - rho * ((np.asarray(B, dtype=np.float64) @ yg_col) @ step_col.T)
                            + rho * (step_col @ step_col.T)
                        )

                    score_hist[i] = float(trial["score"])
                    lsp = ilsp = ilsp + step
                    converged = True
                    score_scale = _bfgs_score_scale(
                        score_type,
                        trial["score"],
                        scale_est=trial["scale_est"],
                        model=getattr(objective, "model", None),
                    )
                    uconv = np.abs(np.asarray(trial["grad"], dtype=np.float64)) > (
                        score_scale * conv_tol
                    )
                    if np.any(uconv):
                        converged = False
                    uconv = np.abs(np.asarray(trial["grad"], dtype=np.float64)) > (
                        score_scale * conv_tol * 0.1
                    )
                    uconv = _apply_dvkk_signal(uconv, trial["dVkk"], 0.1)
                    if abs(float(initial["score"]) - float(trial["score"])) > (
                        score_scale * conv_tol
                    ):
                        if not np.any(uconv):
                            uconv = np.ones_like(uconv, dtype=bool)
                        converged = False

                if converged:
                    if np.all(uconv) or rolled_back:
                        break
                    rolled_back = True
                    counter = 0
                    uconv0 = uconv.copy()
                    while np.any(~uconv0) and counter < 5:
                        lsp[~uconv0] = (
                            lsp[~uconv0] * 0.8 + initial_lsp[~uconv0] * 0.2
                        )
                        (
                            score_rb,
                            grad_rb,
                            _,
                            dvkk_rb,
                            coef_rb,
                            eta_rb,
                            mu_rb,
                            scale_rb,
                        ) = _eval_objective_at(
                            objective,
                            lsp,
                            start_coef=trial["start"],
                            start_eta=trial["eta"],
                            start_mu=trial["mustart"],
                            need_grad=True,
                            commit_start=False,
                        )
                        trial["score"] = float(score_rb)
                        trial["grad"] = np.asarray(grad_rb, dtype=np.float64)
                        trial["dscore"] = float(trial["grad"] @ step)
                        trial["dVkk"] = np.asarray(dvkk_rb, dtype=np.float64)
                        trial["start"] = _copy_state_vector(coef_rb)
                        trial["eta"] = _copy_state_vector(eta_rb)
                        trial["mustart"] = _copy_state_vector(mu_rb)
                        trial["scale_est"] = scale_rb
                        counter += 1
                        uconv0 = np.abs(np.asarray(trial["grad"], dtype=np.float64)) > (
                            score_scale * conv_tol * 20.0
                        )
                        uconv0 = _apply_dvkk_signal(uconv0, trial["dVkk"], 20.0)
                        uconv0 = uconv0 | uconv
                    uconv = np.ones_like(uconv, dtype=bool)
                    ilsp = lsp.copy()
            step_norm = 0.0
            if len(iter_trace) > 0:
                prev_lsp = np.asarray(iter_trace[-1]["log_sp"], dtype=np.float64)
                step_norm = float(np.linalg.norm(ilsp - prev_lsp))
            iter_trace.append(
                {
                    "iter": int(i),
                    "log_sp": np.asarray(ilsp, dtype=np.float64).copy(),
                    "criterion": float(trial["score"]),
                    "gradient": np.asarray(trial["grad"], dtype=np.float64).copy(),
                    "hessian": None,
                    "accepted_step_norm": step_norm,
                    "rank_info": {
                        "source": "outer_bfgs_strict",
                        "line_search_alpha": float(trial["alpha"]),
                        "converged_here": bool(converged),
                        "rolled_back": bool(rolled_back),
                    },
                }
            )
            initial = trial.copy()
            initial["alpha"] = 0.0

        if trial is None:
            ct = "step failed"
            lsp = ilsp.copy()
            trial = initial.copy()
        elif i == int(max_step):
            ct = "iteration limit reached"
        else:
            ct = "full convergence"

        score_f, grad_f, _, _, coef_f, eta_f, mu_f, _ = _eval_objective_at(
            objective,
            lsp,
            start_coef=trial["start"],
            start_eta=trial["eta"],
            start_mu=trial["mustart"],
            need_grad=True,
            commit_start=True,
        )
        if coef_f is not None:
            trial["start"] = _copy_state_vector(coef_f)
        if eta_f is not None:
            trial["eta"] = _copy_state_vector(eta_f)
        if mu_f is not None:
            trial["mustart"] = _copy_state_vector(mu_f)
        hess_approx = _invert_inverse_hessian(B)
        success = ct == "full convergence"
        result = OptimizeResult(
            x=np.asarray(lsp, dtype=np.float64),
            fun=float(score_f),
            jac=np.asarray(grad_f, dtype=np.float64),
            hess=np.asarray(hess_approx, dtype=np.float64),
            success=bool(success),
            status=0 if success else 1,
            message=ct,
            nit=int(i),
            nfev=int(objective.n_fun),
            njev=int(objective.n_jac),
            nhev=int(objective.n_hess),
        )
        # `mgcv::bfgs()` returns the quasi-Newton Hessian approximation, not the
        # exact outer criterion Hessian. Preserve that payload through driver
        # post-processing instead of overwriting it with `criterion_hessian(...)`.
        result.strict_outer_derivatives = True
        result.strict_score_hist = [v for v in score_hist.tolist() if np.isfinite(v)]
        result.optim_trace = [dict(row) for row in iter_trace]
        result.nit = int(len(result.optim_trace) + 1)
        result.outer_info = {
            "conv": ct,
            "iter": int(len(result.optim_trace) + 1),
            "score_hist": result.strict_score_hist,
            "grad": np.asarray(grad_f, dtype=np.float64),
            "hess": np.asarray(hess_approx, dtype=np.float64),
            "convergence": int(result.status),
            "message": str(ct),
            "counts": np.asarray(
                [int(objective.n_fun), int(objective.n_jac)],
                dtype=np.int64,
            ),
        }
        return result
    finally:
        if model is not None and prev_irls_tol is not None:
            model.irls_tol = prev_irls_tol
