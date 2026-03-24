"""
Penalized IRLS smoothness-selection criteria: GCV, UBRE, and Laplace ML/REML.

Functions here solve the penalized system via P-IRLS at each criterion evaluation
and compute the corresponding smoothness-selection score.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .laplace import _ensure_penalty_reparameterization, _laplace_lambda_vector
from .penalty import _static_fixed_and_random_designs

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
    saturated_loglik = float(
        model.family.saturated_loglik(
            y,
            weights=np.ones_like(y, dtype=np.float64),
            n=len(y),
            scale=scale,
        )
    )
    base_objective = (float(sol["deviance"]) + penalty_quad) / (2.0 * scale) - saturated_loglik

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
            objective -= 0.5 * p * np.log(2.0 * np.pi * scale)
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
        objective -= 0.5 * p * np.log(2.0 * np.pi * scale)
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
            objective -= 0.5 * p * np.log(2.0 * np.pi * scale)
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
        objective -= 0.5 * p * np.log(2.0 * np.pi * scale)
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

