"""
Penalized IRLS smoothing-selection criteria: GCV, UBRE, and Laplace ML/REML.

Functions here solve the penalized system via P-IRLS at each criterion evaluation
and compute the corresponding smoothing-selection score.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .laplace import _ensure_penalty_reparameterization, _laplace_lambda_vector
from .penalty import (
    _stable_penalty_logdet,
    _static_fixed_and_random_designs,
    _static_penalty_null_dim,
)


def _gamma_profile_objective_curvature(model, y, Dp, phi, mp, *, method):
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf, np.nan, np.nan
    ls = float(
        model.family.saturated_loglik(
            y,
            weights=np.ones_like(y, dtype=np.float64),
            n=len(y),
            scale=phi,
        )
    )
    from .pirls_deriv import _gamma_saturated_loglik_scale_derivatives

    ls1, ls2 = _gamma_saturated_loglik_scale_derivatives(y, phi)
    reml_ind = 1.0 if method == "REML" else 0.0
    score_lphi = -Dp / (2.0 * phi) - ls1 * phi - 0.5 * mp * reml_ind
    curv_lphi = Dp / (2.0 * phi) - ls2 * (phi ** 2) - ls1 * phi
    return ls, score_lphi, curv_lphi


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

def criterion_gcv_pirls(model, y, log_sp):
    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)
    n = model.n_samples_
    den = 1.0 - model.score_gamma * sol["trace_H"] / n
    if den <= 1e-12 or not np.isfinite(den):
        return np.inf
    return (sol["deviance"] / n) / (den ** 2)


def criterion_ubre_pirls(model, y, log_sp):
    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)
    scale = model.family.known_scale
    if scale is None:
        raise ValueError(
            f"UBRE/AIC requested for family={model.family.name!r}, "
            "but the family does not have known scale."
        )
    n = model.n_samples_
    edf = sol["trace_H"]
    return (sol["deviance"] / n) - scale + (2.0 * model.score_gamma * scale * edf / n)


def _ensure_penalty_reparameterization(model):
    if (
        model.X_fix_ is None
        or model.Z_rand_ is None
        or getattr(model, "_reparam_sp_groups_", None) is None
    ):
        model._build_penalty_reparameterized_system()


def _laplace_lambda_vector(model, sp):
    blocks = getattr(model, "_reparam_rand_blocks_", None)
    if not blocks:
        return np.empty((0,), dtype=np.float64)
    lam_parts = []
    for block in blocks:
        n_pen = int(block["n_pen"])
        if n_pen == 0:
            continue
        sp_val = float(sp[int(block["smoothing_index"])])
        scaling = float(block.get("lambda_scaling", 1.0))
        lam_val = sp_val * scaling
        lam_parts.append(np.full(n_pen, lam_val, dtype=np.float64))
    return np.concatenate(lam_parts) if lam_parts else np.empty((0,), dtype=np.float64)


def _lambda_group_indices(model):
    groups = getattr(model, "_reparam_sp_groups_", None)
    if groups is None:
        return {}
    return {
        int(sp_idx): np.asarray(idxs, dtype=np.int64)
        for sp_idx, idxs in groups.items()
    }


def _penalty_derivative_matrices(model, sp):
    n_full = int(model.n_coef_ + (1 if model.fit_intercept else 0))
    offset0 = 1 if model.fit_intercept else 0
    mats = [
        np.zeros((n_full, n_full), dtype=np.float64)
        for _ in range(int(model.n_smoothing_params_ or 0))
    ]
    if not mats:
        return mats

    for pb in model.penalty_blocks_:
        k = int(pb.smoothing_index)
        sl = pb.coef_slice
        full_sl = slice(offset0 + sl.start, offset0 + sl.stop)
        mats[k][full_sl, full_sl] += float(sp[k]) * np.asarray(pb.matrix, dtype=np.float64)
    return mats


def _pirls_laplace_logdet_term(model, sol, sp, method):
    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    W = np.asarray(sol["working_weights"], dtype=np.float64)

    if q == 0:
        if method == "ML" or p == 0:
            return 0.0

        XtWX_fix = Xf.T @ (W[:, None] * Xf)
        cFix, _ = cho_factor(XtWX_fix, check_finite=False)
        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        return 0.5 * logdet_fix

    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.inf

    ZtW = Zr.T * W[np.newaxis, :]
    M = ZtW @ Zr + np.diag(lam_vec)
    cM, loM = cho_factor(M, check_finite=False)

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    logdet_Lam = float(np.sum(np.log(lam_vec)))
    det_term = 0.5 * (logdet_M - logdet_Lam)

    if method == "ML" or p == 0:
        return det_term

    ZTWX = Zr.T @ (W[:, None] * Xf)
    Minv_ZTWX = cho_solve((cM, loM), ZTWX, check_finite=False)
    XtKX = Xf.T @ (W[:, None] * Xf) - ZTWX.T @ Minv_ZTWX
    cXKX, _ = cho_factor(XtKX, check_finite=False)
    logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return det_term + 0.5 * logdet_XtKX


def _pirls_tensor_coefficient_space_logdet_term(model, sol, sp):
    """Coefficient-space REML/ML determinant term for tensor PIRLS fits.

    `mgcv` evaluates tensor-product REML penalties against the weighted coefficient-space
    system `X'WX + S`. For non-Gaussian tensor terms, the static mixed-model block
    decomposition used by the exact PIRLS Laplace path is not equivalent enough
    numerically, even though the fixed-sp fitted functions agree.
    """
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
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the PIRLS Laplace ML/REML path."
        )

    if not model._can_use_simple_ml_reml_structure():
        raise NotImplementedError(
            "Current PIRLS Laplace ML/REML is available only for terms with "
            "disjoint-support primary smooth penalties, plus at most one "
            "null-space penalty per support block."
        )

    _ensure_penalty_reparameterization(model)

    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)

    scale = float(sol["scale"])
    if not np.isfinite(scale) or scale <= 0:
        return np.inf

    penalty_quad = float(sol["penalty_quadratic"] or 0.0)
    mp = float(
        _static_penalty_null_dim(model)
        + int(bool(getattr(model, "fit_intercept", False)))
    )
    n_obs = float(len(y))
    family_name = str(getattr(model.family, "name", "")).lower()

    if getattr(model.family, "known_scale", None) is None:
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
            base_objective = Dp / (2.0 * phi) - saturated_loglik
        else:
            var = np.clip(np.asarray(model.family.variance(sol["mu"]), dtype=np.float64), 1e-14, None)
            pearson = float(np.sum((np.asarray(y, dtype=np.float64) - np.asarray(sol["mu"], dtype=np.float64)) ** 2 / var))
            denom = n_obs - mp
            if not np.isfinite(denom) or denom <= 0.0:
                return np.inf
            phi = pearson / denom
            if not np.isfinite(phi) or phi <= 0.0:
                return np.inf

            saturated_loglik = float(
                model.family.saturated_loglik(
                    y,
                    weights=np.ones_like(y, dtype=np.float64),
                    n=len(y),
                    scale=phi,
                )
            )
            base_objective = (float(sol["deviance"]) + penalty_quad) / (2.0 * phi) - saturated_loglik
        try:
            det_term = (
                _pirls_tensor_coefficient_space_logdet_term(model, sol, sp)
                if model._has_tensor_terms()
                else _pirls_laplace_logdet_term(model, sol, sp, method)
            )
        except np.linalg.LinAlgError:
            return np.inf
        if not np.isfinite(det_term):
            return np.inf
        objective = base_objective + det_term
        if method == "REML":
            objective -= 0.5 * mp * np.log(2.0 * np.pi * phi)
        return objective

    saturated_loglik = float(
        model.family.saturated_loglik(
            y,
            weights=np.ones_like(y, dtype=np.float64),
            n=len(y),
            scale=scale,
        )
    )
    base_objective = (float(sol["deviance"]) + penalty_quad) / (2.0 * scale) - saturated_loglik

    if model._has_tensor_terms():
        det_term = _pirls_tensor_coefficient_space_logdet_term(model, sol, sp)
        if not np.isfinite(det_term):
            return np.inf
        objective = base_objective + det_term
        if method == "REML":
            objective -= 0.5 * mp * np.log(2.0 * np.pi * scale)
        return objective

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
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
            objective -= 0.5 * mp * np.log(2.0 * np.pi * scale)
        return objective

    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.inf

    ZtW = Zr.T * W[np.newaxis, :]
    M = ZtW @ Zr + np.diag(lam_vec)

    try:
        cM, loM = cho_factor(M, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    logdet_Lam = float(np.sum(np.log(lam_vec)))

    objective = base_objective + 0.5 * (logdet_M - logdet_Lam)

    if method == "ML":
        return objective

    if p > 0:
        objective -= 0.5 * mp * np.log(2.0 * np.pi * scale)
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
        raise NotImplementedError("Joint PIRLS Gamma outer objective is implemented only for family='gamma'.")

    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)

    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf

    penalty_quad = float(sol["penalty_quadratic"] or 0.0)
    mp = float(
        _static_penalty_null_dim(model)
        + int(bool(getattr(model, "fit_intercept", False)))
    )
    Dp = float(sol["deviance"]) + penalty_quad
    saturated_loglik, _, _ = _gamma_profile_objective_curvature(
        model, y, Dp, phi, mp, method=method
    )
    base_objective = Dp / (2.0 * phi) - saturated_loglik
    try:
        det_term = (
            _pirls_tensor_coefficient_space_logdet_term(model, sol, sp)
            if model._has_tensor_terms()
            else _pirls_laplace_logdet_term(model, sol, sp, method)
        )
    except np.linalg.LinAlgError:
        return np.inf
    if not np.isfinite(det_term):
        return np.inf

    objective = base_objective + det_term
    if method == "REML":
        objective -= 0.5 * mp * np.log(2.0 * np.pi * phi)
    return float(objective)


def criterion_ml_reml_pirls_dynamic(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the dynamic PIRLS Laplace ML/REML path."
        )

    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)

    scale = float(sol["scale"])
    if not np.isfinite(scale) or scale <= 0:
        return np.inf

    saturated_loglik = float(
        model.family.saturated_loglik(
            y,
            weights=np.ones_like(y, dtype=np.float64),
            n=len(y),
            scale=scale,
        )
    )
    base_objective = (
        (float(sol["deviance"]) + float(sol["penalty_quadratic"] or 0.0)) / (2.0 * scale)
        - saturated_loglik
    )

    X = np.asarray(sol["X"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    mp = float(
        _static_penalty_null_dim(model)
        + int(bool(getattr(model, "fit_intercept", False)))
    )
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
            objective -= 0.5 * mp * np.log(2.0 * np.pi * scale)
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
        objective -= 0.5 * mp * np.log(2.0 * np.pi * scale)
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
