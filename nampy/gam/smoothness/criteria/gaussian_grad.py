"""Exact gradients for Gaussian REML in the Laplace reparameterization."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .laplace import _lambda_group_indices, _laplace_lambda_vector

def criterion_gradient_ml_reml_exact(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the exact Gaussian ML/REML/LAML gradient path."
        )

    if not model._can_use_exact_gaussian_ml_reml():
        raise NotImplementedError(
            "Exact Gaussian ML/REML/LAML gradients are currently available only for "
            "terms with one primary smooth penalty, plus at most one "
            "null-space penalty."
        )

    free_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    if int(np.sum(free_mask)) == 0:
        return np.empty((0,), dtype=np.float64)

    y = model.family.validate_y(y)
    y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
    sp = model._expand_smoothing_params_from_log(log_sp)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    n = Xf.shape[0]
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)

    grad_full = np.zeros(int(model.n_smoothing_params_), dtype=np.float64)

    if q == 0:
        return grad_full[free_mask]

    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.full(int(np.sum(free_mask)), np.nan, dtype=np.float64)

    groups = _lambda_group_indices(model)
    M = model.ZtZ_rand_ + np.diag(lam_vec)
    try:
        cM, loM = cho_factor(M, check_finite=False)
    except np.linalg.LinAlgError:
        return np.full(int(np.sum(free_mask)), np.nan, dtype=np.float64)

    ZTy = Zr.T @ y_eff
    Minv_ZTy = cho_solve((cM, loM), ZTy, check_finite=False)
    Ky = y_eff - Zr @ Minv_ZTy

    if p > 0:
        ZTX = Zr.T @ Xf
        Minv_ZTX = cho_solve((cM, loM), ZTX, check_finite=False)
        KX = Xf - Zr @ Minv_ZTX
        XtKX = Xf.T @ KX
        try:
            cXKX, loXKX = cho_factor(XtKX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.full(int(np.sum(free_mask)), np.nan, dtype=np.float64)

        XtKy = Xf.T @ Ky
        b_hat = cho_solve((cXKX, loXKX), XtKy, check_finite=False)
        e = y_eff - Xf @ b_hat
        rss_v = max(float(y_eff @ Ky - XtKy @ b_hat), 1e-14)
    else:
        cXKX = loXKX = None
        e = y_eff
        rss_v = max(float(y_eff @ Ky), 1e-14)

    eye_q = np.eye(q, dtype=np.float64)

    for sp_idx, group in groups.items():
        if group.size == 0:
            continue

        lam = float(sp[sp_idx])
        Minv_cols = cho_solve((cM, loM), eye_q[:, group], check_finite=False)
        U = Zr @ Minv_cols

        uTe = U.T @ e
        drss = lam * float(uTe @ uTe)

        trace_block = float(np.trace(Minv_cols[group, :]))
        d_logdet_vtilde = lam * trace_block - float(group.size)

        if method == "ML":
            grad_full[sp_idx] = (n * drss / rss_v) + d_logdet_vtilde
            continue

        d_logdet_xtkx = 0.0
        if p > 0:
            B = Xf.T @ U
            CinvB = cho_solve((cXKX, loXKX), B, check_finite=False)
            d_logdet_xtkx = lam * float(np.sum(B * CinvB))

        grad_full[sp_idx] = ((n - p) * drss / rss_v) + d_logdet_vtilde + d_logdet_xtkx

    return grad_full[free_mask]
