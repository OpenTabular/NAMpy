"""
Penalized IRLS smoothing-selection criteria: GCV, UBRE, and Laplace ML/REML.

Functions here solve the penalized system via P-IRLS at each criterion evaluation
and compute the corresponding smoothing-selection score.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .....model_state import _coef_column_offset
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import (
    _stable_penalty_logdet,
    _static_fixed_and_random_designs,
    _static_penalty_null_dim,
    can_use_simple_ml_reml_structure,
    dynamic_reparam_design,
)
from .common import _prior_weights


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
            from .family_gamma import (
                _gamma_profile_objective_curvature,
                _solve_gamma_profile_scale,
            )

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
        if method in {"REML", "LAML"}:
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
