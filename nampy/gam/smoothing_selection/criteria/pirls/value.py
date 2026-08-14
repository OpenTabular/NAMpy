"""
Penalized IRLS smoothing-selection criteria: GCV, UBRE, and Laplace ML/REML.

Functions here solve the penalized system via P-IRLS at each criterion evaluation
and compute the corresponding smoothing-selection score.
"""

from contextlib import contextmanager

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...._model_state import _coef_column_offset
from ....fit.backends import solve_pirls_given_smoothing
from ....fit.capabilities import can_use_simple_ml_reml_structure
from ....fit.smoothing_params import expand_smoothing_params_from_log
from ...reparam import (
    _stable_penalty_logdet,
    _static_fixed_and_random_designs,
    _static_penalty_null_dim,
    dynamic_reparam_design,
)


def _is_joint_negbin_theta_model(model) -> bool:
    return str(getattr(model.family, "name", "")).lower() == "negbin" and bool(
        getattr(model.family, "estimate_theta", False)
    )


def _copy_state_vector(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def _current_joint_negbin_eval_state(model):
    result = getattr(model, "_optim_result", None)
    result_state = (
        None if result is None else getattr(result, "joint_negbin_state", None)
    )
    if isinstance(result_state, dict):
        return {
            "coef": _copy_state_vector(result_state.get("coef", None)),
            "eta": _copy_state_vector(result_state.get("eta", None)),
            "mu": _copy_state_vector(result_state.get("mu", None)),
            "theta": float(result_state.get("theta", model.family.getTheta(True))),
        }

    baseline = getattr(model, "_joint_negbin_fd_baseline_", None)
    if isinstance(baseline, dict):
        return {
            "coef": _copy_state_vector(baseline.get("coef", None)),
            "eta": _copy_state_vector(baseline.get("eta", None)),
            "mu": _copy_state_vector(baseline.get("mu", None)),
            "theta": float(baseline.get("theta", model.family.getTheta(True))),
        }

    return {
        "coef": _copy_state_vector(
            getattr(
                model, "_pirls_coef_start_", getattr(model, "_pirls_last_coef_", None)
            )
        ),
        "eta": _copy_state_vector(
            getattr(
                model, "_pirls_eta_start_", getattr(model, "_pirls_last_eta_", None)
            )
        ),
        "mu": _copy_state_vector(
            getattr(model, "_pirls_mu_start_", getattr(model, "_pirls_last_mu_", None))
        ),
        "theta": float(model.family.getTheta(True)),
    }


@contextmanager
def _frozen_joint_negbin_eval_state(model, baseline_state=None):
    prev = {
        "eval_coef": _copy_state_vector(getattr(model, "_pirls_eval_start_", None)),
        "eval_eta": _copy_state_vector(getattr(model, "_pirls_eval_eta_start_", None)),
        "eval_mu": _copy_state_vector(getattr(model, "_pirls_eval_mu_start_", None)),
        "lock": bool(getattr(model, "_pirls_lock_start_", False)),
        "coef": _copy_state_vector(getattr(model, "_pirls_coef_start_", None)),
        "eta": _copy_state_vector(getattr(model, "_pirls_eta_start_", None)),
        "mu": _copy_state_vector(getattr(model, "_pirls_mu_start_", None)),
        "last_coef": _copy_state_vector(getattr(model, "_pirls_last_coef_", None)),
        "last_eta": _copy_state_vector(getattr(model, "_pirls_last_eta_", None)),
        "last_mu": _copy_state_vector(getattr(model, "_pirls_last_mu_", None)),
        "theta": float(model.family.getTheta(True)),
    }
    state = (
        _current_joint_negbin_eval_state(model)
        if baseline_state is None
        else {
            "coef": _copy_state_vector(baseline_state.get("coef", None)),
            "eta": _copy_state_vector(baseline_state.get("eta", None)),
            "mu": _copy_state_vector(baseline_state.get("mu", None)),
            "theta": float(baseline_state.get("theta", model.family.getTheta(True))),
        }
    )
    model._pirls_eval_start_ = _copy_state_vector(state.get("coef", None))
    model._pirls_eval_eta_start_ = _copy_state_vector(state.get("eta", None))
    model._pirls_eval_mu_start_ = _copy_state_vector(state.get("mu", None))
    model._pirls_lock_start_ = True
    model.family.putTheta(float(np.log(state["theta"])))
    try:
        yield state
    finally:
        model._pirls_eval_start_ = prev["eval_coef"]
        model._pirls_eval_eta_start_ = prev["eval_eta"]
        model._pirls_eval_mu_start_ = prev["eval_mu"]
        model._pirls_lock_start_ = prev["lock"]
        model._pirls_coef_start_ = prev["coef"]
        model._pirls_eta_start_ = prev["eta"]
        model._pirls_mu_start_ = prev["mu"]
        model._pirls_last_coef_ = prev["last_coef"]
        model._pirls_last_eta_ = prev["last_eta"]
        model._pirls_last_mu_ = prev["last_mu"]
        model.family.putTheta(float(np.log(prev["theta"])))


def criterion_ml_reml_pirls_frozen_negbin(
    model, y, log_sp, method, baseline_state=None
):
    with _frozen_joint_negbin_eval_state(model, baseline_state=baseline_state):
        return criterion_ml_reml_pirls(model, y, log_sp, method)


def _gamma_profile_objective_curvature(model, y, Dp, phi, mp, *, method):
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf, np.nan, np.nan
    weights = getattr(model, "prior_weights_", None)
    if weights is None:
        weights = np.ones_like(np.asarray(y, dtype=np.float64), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    ls = np.asarray(model.family.ls(y, weights, len(y), phi), dtype=np.float64)
    nobs = float(len(y))
    n_true = getattr(model, "n_true_", None)
    if n_true is None:
        n_true = nobs
    n_true = float(n_true)
    if not np.isfinite(n_true) or n_true <= 0.0 or not np.isfinite(nobs) or nobs <= 0.0:
        fac = 1.0
    else:
        fac = n_true / nobs
    ls *= fac
    reml_ind = 1.0 if method == "REML" else 0.0
    gamma = float(model.score_gamma)
    score_lphi = (-Dp / (2.0 * phi) - ls[1] * phi) / gamma - 0.5 * mp * reml_ind
    curv_lphi = (Dp / (2.0 * phi) - ls[2] * (phi**2) - ls[1] * phi) / gamma
    return float(ls[0]), float(score_lphi), float(curv_lphi)


def _solve_gamma_profile_scale(model, y, Dp, mp, *, method, init_scale):
    phi = float(max(init_scale, 1e-12))
    if not np.isfinite(phi) or phi <= 0.0:
        phi = max(float(Dp) / max(float(len(y)), 1.0), 1e-6)

    for _ in range(40):
        _, score_lphi, curv_lphi = _gamma_profile_objective_curvature(
            model, y, Dp, phi, mp, method=method
        )
        if not np.isfinite(score_lphi) or not np.isfinite(curv_lphi):
            break
        if abs(score_lphi) <= 1e-12:
            return phi
        if abs(curv_lphi) <= 1e-14:
            break
        step = score_lphi / curv_lphi
        if not np.isfinite(step):
            break
        if abs(step) <= 1e-12:
            return phi
        new_phi = float(np.exp(np.log(phi) - step))
        if not np.isfinite(new_phi) or new_phi <= 0.0:
            break
        phi = new_phi
    return phi


def _prior_weights(model, y):
    weights = getattr(model, "prior_weights_", None)
    if weights is None:
        return np.ones_like(np.asarray(y, dtype=np.float64), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def _saturated_loglik(model, y, *, scale):
    y_arr = np.asarray(y, dtype=np.float64)
    weights = _prior_weights(model, y_arr)
    nobs = float(len(y_arr))
    n_true = getattr(model, "n_true_", None)
    if n_true is None:
        fac = 1.0
    else:
        n_true = float(n_true)
        fac = (
            n_true / nobs
            if np.isfinite(n_true) and n_true > 0.0 and nobs > 0.0
            else 1.0
        )
    return float(
        fac
        * model.family.saturated_loglik(
            y_arr,
            weights=weights,
            n=len(y_arr),
            scale=scale,
        )
    )


def criterion_gcv_pirls(model, y, log_sp):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    n = model.n_samples_
    den = 1.0 - model.score_gamma * sol["trace_H"] / n
    if not np.isfinite(den) or den == 0.0:
        return np.inf
    return (sol["deviance"] / n) / (den**2)


def criterion_ubre_pirls(model, y, log_sp):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    scale = model.family.known_scale
    if scale is None:
        raise ValueError(
            f"UBRE/AIC requested for family={model.family.name!r}, "
            "but the family does not have known scale."
        )
    n = model.n_samples_
    edf = sol["trace_H"]
    return (sol["deviance"] / n) - scale + (2.0 * model.score_gamma * scale * edf / n)


def _pirls_laplace_logdet_term(model, sol, sp, method):
    design = dynamic_reparam_design(model, sol["X"], sp)
    Xf = design.X_fix
    Zr = design.Z_rand
    p = int(Xf.shape[1])
    q = int(Zr.shape[1])
    W = np.asarray(sol["working_weights"], dtype=np.float64)

    if q == 0:
        if method == "ML" or p == 0:
            return 0.0

        XtWX_fix = Xf.T @ (W[:, None] * Xf)
        cFix, _ = cho_factor(XtWX_fix, check_finite=False)
        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        return 0.5 * logdet_fix

    ZtW = Zr.T * W[np.newaxis, :]
    M = ZtW @ Zr + np.eye(q, dtype=np.float64)
    cM, loM = cho_factor(M, check_finite=False)

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    det_term = 0.5 * logdet_M

    if method == "ML" or p == 0:
        return det_term

    ZTWX = Zr.T @ (W[:, None] * Xf)
    Minv_ZTWX = cho_solve((cM, loM), ZTWX, check_finite=False)
    XtKX = Xf.T @ (W[:, None] * Xf) - ZTWX.T @ Minv_ZTWX
    cXKX, _ = cho_factor(XtKX, check_finite=False)
    logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return det_term + 0.5 * logdet_XtKX


def _pirls_tensor_coefficient_space_logdet_term(model, y, sol, sp, method):
    """Coefficient-space REML/ML determinant term for tensor PIRLS fits.

    `mgcv` evaluates tensor-product REML penalties against the weighted coefficient-space
    system `X'WX + S`. For non-Gaussian tensor terms, the static mixed-model block
    decomposition used by the exact PIRLS Laplace path is not equivalent enough
    numerically, even though the fixed-sp fitted functions agree.
    """
    if str(method).upper() == "ML":
        from .derivatives import _gdi1_kernel

        return float(_gdi1_kernel(model, y, sol, sp, method="ML").K)

    X = np.asarray(sol["X"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    P = np.asarray(sol["P"], dtype=np.float64)
    A = X.T @ (W[:, None] * X) + P
    try:
        cA, _ = cho_factor(A, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf
    logdet_A = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
    logdet_S = float(_stable_penalty_logdet(model, sp))
    if not np.isfinite(logdet_S):
        return np.inf
    return 0.5 * (logdet_A - logdet_S)


def criterion_ml_reml_pirls(model, y, log_sp, method):
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            "Current PIRLS Laplace ML/REML is unavailable only when "
            "null-space penalties couple disconnected primary support "
            "components."
        )

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)

    return _pirls_ml_reml_objective_from_solution(model, y, sol, sp, method)


def _pirls_ml_reml_objective_from_solution(model, y, sol, sp, method):
    method = str(method).upper()
    scale = float(sol["scale"])
    if not np.isfinite(scale) or scale <= 0:
        return np.inf
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf

    penalty_quad = float(sol["penalty_quadratic"] or 0.0)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_obs = float(len(y))
    family_name = str(getattr(model.family, "name", "")).lower()
    use_exact_logdet = bool(
        can_use_simple_ml_reml_structure(model)
        and not model._has_tensor_terms()
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
    )

    if getattr(model.family, "known_scale", None) is None:
        if family_name == "gaussian":
            # mgcv::gam.fit3 treats the gaussian scale as an extra joint
            # coordinate (mgcv/R/mgcv.r:2025-2033); at the profiled optimum
            # dlr.dlphi = 0 (mgcv/R/gam.fit3.r:628-630 with the gaussian ls
            # from gam.fit3.r:2503-2508) gives the closed form
            # phi = Dp / (n - gamma*Mp*remlInd). Evaluate the joint gam.fit3
            # score there instead of the Pearson/(n - Mp) plug-in, which is
            # mgcv's P-REML scale, not the (P)REML criterion scale.
            try:
                from .derivatives import _gdi1_kernel

                kernel = _gdi1_kernel(model, y, sol, sp, method=method)
            except np.linalg.LinAlgError:
                return np.inf
            Dp = float(sol["deviance"]) + float(kernel.bSb)
            y_arr = np.asarray(y, dtype=np.float64)
            weights = _prior_weights(model, y_arr)
            nobs = float(len(y_arr))
            n_eff = nobs
            n_true = getattr(model, "n_true_", None)
            if n_true is not None:
                n_true = float(n_true)
                if np.isfinite(n_true) and n_true > 0.0:
                    n_eff = n_true
            reml_ind = 1.0 if method == "REML" else 0.0
            denom = n_eff - gamma * mp * reml_ind
            if not np.isfinite(denom) or denom <= 0.0:
                return np.inf
            phi = Dp / denom
            if not np.isfinite(phi) or phi <= 0.0:
                return np.inf
            ls = np.asarray(
                model.family.ls(y_arr, weights, len(y_arr), phi),
                dtype=np.float64,
            )
            if n_eff != nobs and nobs > 0.0:
                ls *= n_eff / nobs
            objective = (Dp / (2.0 * phi) - float(ls[0])) / gamma + float(
                kernel.K
            )
            if method == "REML":
                objective -= 0.5 * mp * (
                    np.log(2.0 * np.pi * phi) - np.log(gamma)
                )
            return float(objective)
        if family_name == "gamma":
            Dp = float(sol["deviance"]) + penalty_quad
            phi = _solve_gamma_profile_scale(
                model,
                y,
                Dp,
                mp,
                method=method,
                init_scale=float(sol["scale"]),
            )
            if not np.isfinite(phi) or phi <= 0.0:
                return np.inf
            saturated_loglik, _, _ = _gamma_profile_objective_curvature(
                model, y, Dp, phi, mp, method=method
            )
            base_objective = Dp / (2.0 * phi * gamma) - saturated_loglik / gamma
        else:
            var = np.clip(
                np.asarray(model.family.variance(sol["mu"]), dtype=np.float64),
                1e-14,
                None,
            )
            weights = _prior_weights(model, y)
            pearson = float(
                np.sum(
                    weights
                    * (
                        (
                            np.asarray(y, dtype=np.float64)
                            - np.asarray(sol["mu"], dtype=np.float64)
                        )
                        ** 2
                        / var
                    )
                )
            )
            denom = n_obs - mp
            if not np.isfinite(denom) or denom <= 0.0:
                return np.inf
            phi = pearson / denom
            if not np.isfinite(phi) or phi <= 0.0:
                return np.inf

            saturated_loglik = _saturated_loglik(model, y, scale=phi)
            base_objective = (float(sol["deviance"]) + penalty_quad) / (
                2.0 * phi * gamma
            ) - saturated_loglik / gamma
        try:
            if use_exact_logdet:
                from .derivatives import _gdi1_kernel

                det_term = float(_gdi1_kernel(model, y, sol, sp, method=method).K)
            else:
                det_term = (
                    _pirls_tensor_coefficient_space_logdet_term(
                        model, y, sol, sp, method
                    )
                    if model._has_tensor_terms()
                    else _pirls_laplace_logdet_term(model, sol, sp, method)
                )
        except np.linalg.LinAlgError:
            return np.inf
        if not np.isfinite(det_term):
            return np.inf
        objective = base_objective + det_term
        if method == "REML":
            objective -= 0.5 * mp * (np.log(2.0 * np.pi * phi) - np.log(gamma))
        return objective

    saturated_loglik = _saturated_loglik(model, y, scale=scale)
    base_objective = (float(sol["deviance"]) + penalty_quad) / (
        2.0 * scale * gamma
    ) - saturated_loglik / gamma

    if model._has_tensor_terms():
        det_term = _pirls_tensor_coefficient_space_logdet_term(
            model, y, sol, sp, method
        )
        if not np.isfinite(det_term):
            return np.inf
        objective = base_objective + det_term
        if method == "REML":
            objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
        return objective

    if use_exact_logdet:
        from .derivatives import _gdi1_kernel

        try:
            det_term = float(_gdi1_kernel(model, y, sol, sp, method=method).K)
        except np.linalg.LinAlgError:
            return np.inf
        if not np.isfinite(det_term):
            return np.inf
        objective = base_objective + det_term
        if method == "REML":
            objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
        return objective

    design = dynamic_reparam_design(model, sol["X"], sp)
    Xf = design.X_fix
    Zr = design.Z_rand
    p = int(Xf.shape[1])
    q = int(Zr.shape[1])
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    if q == 0:
        if method == "ML":
            return base_objective

        if p == 0:
            return base_objective

        XtWX_fix = Xf.T @ (W[:, None] * Xf)
        try:
            cFix, _ = cho_factor(XtWX_fix, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        objective = base_objective + 0.5 * logdet_fix
        if method == "REML":
            objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
        return objective

    ZtW = Zr.T * W[np.newaxis, :]
    M = ZtW @ Zr + np.eye(q, dtype=np.float64)

    try:
        cM, loM = cho_factor(M, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    objective = base_objective + 0.5 * logdet_M

    if method == "ML":
        return objective

    if p > 0:
        objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
    else:
        return objective

    ZTWX = Zr.T @ (W[:, None] * Xf)
    Minv_ZTWX = cho_solve((cM, loM), ZTWX, check_finite=False)
    XtKX = Xf.T @ (W[:, None] * Xf) - ZTWX.T @ Minv_ZTWX

    try:
        cXKX, _ = cho_factor(XtKX, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return objective + 0.5 * logdet_XtKX


def criterion_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    method = str(method).upper()
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError(
            "Joint PIRLS Gamma outer objective is implemented only for family='gamma'."
        )

    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)

    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    use_exact_logdet = bool(
        can_use_simple_ml_reml_structure(model)
        and not model._has_tensor_terms()
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
    )
    try:
        if use_exact_logdet:
            from .derivatives import _gdi1_kernel

            kernel = _gdi1_kernel(model, y, sol, sp, method=method)
            Dp = float(sol["deviance"]) + float(kernel.bSb)
            det_term = float(kernel.K)
        else:
            penalty_quad = float(sol["penalty_quadratic"] or 0.0)
            Dp = float(sol["deviance"]) + penalty_quad
            det_term = (
                _pirls_tensor_coefficient_space_logdet_term(model, y, sol, sp, method)
                if model._has_tensor_terms()
                else _pirls_laplace_logdet_term(model, sol, sp, method)
            )
    except np.linalg.LinAlgError:
        return np.inf
    if not np.isfinite(det_term):
        return np.inf

    saturated_loglik, _, _ = _gamma_profile_objective_curvature(
        model, y, Dp, phi, mp, method=method
    )
    base_objective = Dp / (2.0 * phi * gamma) - saturated_loglik / gamma
    objective = base_objective + det_term
    if method == "REML":
        objective -= 0.5 * mp * (np.log(2.0 * np.pi * phi) - np.log(gamma))
    return float(objective)


def criterion_ml_reml_pirls_gaussian_joint(model, y, log_sp, log_phi, method):
    """Joint Gaussian PIRLS ML/REML criterion from ``mgcv::gam.fit3``."""
    method = str(method).upper()
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gaussian":
        raise NotImplementedError(
            "Joint PIRLS Gaussian outer objective is implemented only for "
            "family='gaussian'."
        )
    if bool(getattr(model.family, "supports_closed_form_solve", False)):
        raise NotImplementedError(
            "The PIRLS Gaussian joint objective is only for noncanonical links."
        )

    phi = float(np.exp(float(log_phi)))
    gamma = float(model.score_gamma)
    if (
        not np.isfinite(phi)
        or phi <= 0.0
        or not np.isfinite(gamma)
        or gamma <= 0.0
    ):
        return np.inf

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
    try:
        from .derivatives import _gdi1_kernel

        kernel = _gdi1_kernel(model, y, sol, sp, method=method)
    except np.linalg.LinAlgError:
        return np.inf

    y_arr = np.asarray(y, dtype=np.float64)
    weights = _prior_weights(model, y_arr)
    ls = np.asarray(
        model.family.ls(y_arr, weights, len(y_arr), phi), dtype=np.float64
    )
    nobs = float(len(y_arr))
    n_true = getattr(model, "n_true_", None)
    if n_true is not None:
        n_true = float(n_true)
        if np.isfinite(n_true) and n_true > 0.0 and nobs > 0.0:
            ls *= n_true / nobs

    dp = float(sol["deviance"]) + float(kernel.bSb)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    reml_ind = 1.0 if method == "REML" else 0.0
    objective = (dp / (2.0 * phi) - float(ls[0])) / gamma + float(kernel.K)
    objective -= reml_ind * 0.5 * mp * (
        np.log(2.0 * np.pi * phi) - np.log(gamma)
    )
    return float(objective)


def criterion_ml_reml_pirls_negbin_joint(model, y, log_sp, log_theta, method):
    """Joint (log_sp, log_theta) NB REML outer objective.

    Sets family `log(theta)` as EFS initialization, then
    evaluates the PIRLS REML criterion.  EFS (Embedded Fisher Scoring) updates
    theta after each inner IRLS step inside :func:`irls_core`, mirroring
    mgcv's ``gam.fit4.r`` extended-family pattern where log(theta) is prepended
    to ``lsp`` and theta is updated per PIRLS step within the inner loop.
    """
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "negbin":
        raise NotImplementedError(
            "Joint PIRLS NegBin outer objective is implemented only for family='negbin'."
        )
    theta = float(np.exp(float(log_theta)))
    if not np.isfinite(theta) or theta <= 0.0:
        return np.inf
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        return criterion_ml_reml_pirls(model, y, log_sp, method)
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)


def criterion_ml_reml_pirls_dynamic(model, y, log_sp, method):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)

    scale = float(sol["scale"])
    if not np.isfinite(scale) or scale <= 0:
        return np.inf
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf

    saturated_loglik = _saturated_loglik(model, y, scale=scale)
    base_objective = (
        float(sol["deviance"]) + float(sol["penalty_quadratic"] or 0.0)
    ) / (2.0 * scale * gamma) - saturated_loglik / gamma

    X = np.asarray(sol["X"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    Xf, Zr, split = _static_fixed_and_random_designs(model, X, sp)
    p = int(Xf.shape[1])
    q = int(Zr.shape[1])

    if q == 0:
        if method == "ML":
            return base_objective

        if p == 0:
            return base_objective

        XtWX_fix = Xf.T @ (W[:, None] * Xf)
        try:
            cFix, _ = cho_factor(XtWX_fix, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        objective = base_objective + 0.5 * logdet_fix
        if method == "REML":
            objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
        return objective

    ZtW = Zr.T * W[np.newaxis, :]
    M = ZtW @ Zr + np.eye(q, dtype=np.float64)
    try:
        cM, loM = cho_factor(M, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    objective = base_objective + 0.5 * (logdet_M - float(split["logdet_plus"]))

    if method == "ML":
        return objective

    if p > 0:
        objective -= 0.5 * mp * (np.log(2.0 * np.pi * scale) - np.log(gamma))
    else:
        return objective

    ZTWX = Zr.T @ (W[:, None] * Xf)
    Minv_ZTWX = cho_solve((cM, loM), ZTWX, check_finite=False)
    XtKX = Xf.T @ (W[:, None] * Xf) - ZTWX.T @ Minv_ZTWX

    try:
        cXKX, _ = cho_factor(XtKX, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return objective + 0.5 * logdet_XtKX
