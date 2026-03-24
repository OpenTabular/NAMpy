"""Exact first/second derivatives of PIRLS Laplace ML/REML criteria."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .pirls_reml_derivative_blocks import (
    _hat_matrix_trace_and_sp_derivatives,
    _logdet_penalized_system_derivatives,
    _penalty_quadratic_and_sp_derivatives,
    _quadratic_form_in_beta_directions,
    _working_weight_derivatives_wrt_linpred,
)
from .laplace import (
    _ensure_penalty_reparameterization,
    _lambda_group_indices,
    _laplace_lambda_vector,
    _penalty_derivative_matrices,
)

def criterion_gradient_ml_reml_pirls_exact(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the exact PIRLS ML/REML gradient path."
        )

    if not model._can_use_simple_ml_reml_structure():
        raise NotImplementedError(
            "Exact PIRLS ML/REML gradients are currently available only for "
            "terms with one primary smooth penalty, plus at most one "
            "null-space penalty."
        )

    if not bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False)):
        raise NotImplementedError(
            f"Family {model.family.name!r} does not yet provide exact PIRLS first derivatives."
        )

    _ensure_penalty_reparameterization(model)

    free_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    if int(np.sum(free_mask)) == 0:
        return np.empty((0,), dtype=np.float64)

    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)

    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    scale = float(sol["scale"])
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dW_deta, _ = _working_weight_derivatives_wrt_linpred(model, y, eta, sol["mu"], W)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    grad_full = np.zeros(int(model.n_smoothing_params_), dtype=np.float64)
    grad_legacy = np.zeros_like(grad_full)
    dA_store = [None] * int(model.n_smoothing_params_)

    dbeta_store = [np.zeros_like(beta, dtype=np.float64) for _ in range(int(model.n_smoothing_params_ or 0))]

    if q == 0:
        for j, Pj in enumerate(P_derivs):
            if np.any(Pj):
                grad_full[j] = 0.5 * float(beta @ (Pj @ beta))
                grad_legacy[j] = grad_full[j]
                dA_store[j] = Pj.copy()
                dbeta_store[j] = -(A_inv @ (Pj @ beta))

        if method == "REML" and p > 0:
            XtWX_fix = Xf.T @ (W[:, None] * Xf)
            cFix, loFix = cho_factor(XtWX_fix, check_finite=False)
            C_inv = cho_solve((cFix, loFix), np.eye(p), check_finite=False)
            for j, Pj in enumerate(P_derivs):
                if not np.any(Pj):
                    continue
                dbeta_j = -(A_inv @ (Pj @ beta))
                dbeta_store[j] = dbeta_j
                dW_j = dW_deta * (X @ dbeta_j)
                dXtWX_j = Xf.T @ (dW_j[:, None] * Xf)
                grad_full[j] += 0.5 * float(np.sum(C_inv * dXtWX_j))
                grad_legacy[j] = grad_full[j]
                dA_store[j] = X.T @ (dW_j[:, None] * X) + Pj

        return grad_full[free_mask]

    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.full(int(np.sum(free_mask)), np.nan, dtype=np.float64)

    groups = _lambda_group_indices(model)
    ZtW = Zr.T * W[np.newaxis, :]
    B = ZtW @ Xf if p > 0 else np.empty((q, 0), dtype=np.float64)
    M = ZtW @ Zr + np.diag(lam_vec)
    cM, loM = cho_factor(M, check_finite=False)
    Minv = cho_solve((cM, loM), np.eye(q), check_finite=False)

    if method == "REML" and p > 0:
        XtKX = Xf.T @ (W[:, None] * Xf) - B.T @ Minv @ B
        cC, loC = cho_factor(XtKX, check_finite=False)
        C_inv = cho_solve((cC, loC), np.eye(p), check_finite=False)
    else:
        C_inv = None

    for j, Pj in enumerate(P_derivs):
        if not np.any(Pj):
            continue

        dbeta_j = -(A_inv @ (Pj @ beta))
        dbeta_store[j] = dbeta_j
        deta_j = X @ dbeta_j
        dW_j = dW_deta * deta_j
        dXtWX_j = X.T @ (dW_j[:, None] * X)

        dM_j = Zr.T @ (dW_j[:, None] * Zr)
        group = groups.get(j)
        if group is not None and group.size > 0:
            dM_j[np.ix_(group, group)] += float(sp[j]) * np.eye(group.size, dtype=np.float64)

        grad_j = 0.5 * float(beta @ (Pj @ beta))
        grad_j += 0.5 * float(np.sum(Minv * dM_j))
        if group is not None and group.size > 0:
            grad_j -= 0.5 * float(group.size)

        if method == "REML" and p > 0:
            G_j = Xf.T @ (dW_j[:, None] * Xf)
            dB_j = Zr.T @ (dW_j[:, None] * Xf)
            dC_j = (
                G_j
                - dB_j.T @ Minv @ B
                - B.T @ Minv @ dB_j
                + B.T @ Minv @ dM_j @ Minv @ B
            )
            grad_j += 0.5 * float(np.sum(C_inv * dC_j))

        grad_legacy[j] = grad_j
        grad_full[j] = grad_j
        dA_store[j] = dXtWX_j + Pj

    return grad_full[free_mask]


def criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the exact PIRLS ML/REML Hessian path."
        )

    if not model._can_use_simple_ml_reml_structure():
        raise NotImplementedError(
            "Exact PIRLS ML/REML Hessians are currently available only for "
            "terms with one primary smooth penalty, plus at most one "
            "null-space penalty."
        )

    if not bool(getattr(model.family, "supports_exact_pirls_second_derivatives", False)):
        raise NotImplementedError(
            f"Family {model.family.name!r} does not yet provide exact PIRLS second derivatives."
        )

    _ensure_penalty_reparameterization(model)

    free_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    free_idx = np.flatnonzero(free_mask)
    n_free = int(free_idx.size)
    if n_free == 0:
        return np.empty((0, 0), dtype=np.float64)

    sp = model._expand_smoothing_params_from_log(log_sp)
    sol = model._solve_pirls_given_smoothing(y, sp)

    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    scale = float(sol["scale"])
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dW_eta, d2W_eta = _working_weight_derivatives_wrt_linpred(model, y, eta, sol["mu"], W)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)

    n_sp = int(model.n_smoothing_params_ or 0)
    dbeta = [None] * n_sp
    deta = [None] * n_sp
    dW = [None] * n_sp
    dA = [None] * n_sp
    dM = [None] * n_sp
    dB = [None] * n_sp
    dG = [None] * n_sp
    dXtWX = [None] * n_sp
    d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
    d2A_mat = [[None] * n_sp for _ in range(n_sp)]
    d2XtWX_mat = [[None] * n_sp for _ in range(n_sp)]

    groups = _lambda_group_indices(model)

    if q > 0:
        ZtW = Zr.T * W[np.newaxis, :]
        B0 = ZtW @ Xf if p > 0 else np.empty((q, 0), dtype=np.float64)
        M = ZtW @ Zr + np.diag(_laplace_lambda_vector(model, sp))
        cM, loM = cho_factor(M, check_finite=False)
        Minv = cho_solve((cM, loM), np.eye(q), check_finite=False)
    else:
        B0 = None
        M = None
        Minv = None

    if method == "REML" and p > 0:
        if q > 0:
            C = Xf.T @ (W[:, None] * Xf) - B0.T @ Minv @ B0
        else:
            C = Xf.T @ (W[:, None] * Xf)
        cC, loC = cho_factor(C, check_finite=False)
        C_inv = cho_solve((cC, loC), np.eye(p), check_finite=False)
    else:
        C = C_inv = None

    for j in range(n_sp):
        Pj = P_derivs[j]
        dbeta_j = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
        deta_j = X @ dbeta_j
        dW_j = dW_eta * deta_j
        dXtWX_j = X.T @ (dW_j[:, None] * X)
        dA_j = dXtWX_j + Pj
        dXtWX[j] = dXtWX_j

        dbeta[j] = dbeta_j
        deta[j] = deta_j
        dW[j] = dW_j
        dA[j] = dA_j

        if q > 0:
            dM_j = Zr.T @ (dW_j[:, None] * Zr)
            group_j = groups.get(j)
            if group_j is not None and group_j.size > 0:
                dM_j[np.ix_(group_j, group_j)] += float(sp[j]) * np.eye(group_j.size, dtype=np.float64)
            dM[j] = dM_j
            if p > 0:
                dB[j] = Zr.T @ (dW_j[:, None] * Xf)
                dG[j] = Xf.T @ (dW_j[:, None] * Xf)
            else:
                dB[j] = np.empty((q, 0), dtype=np.float64)
                dG[j] = np.empty((0, 0), dtype=np.float64)
        elif p > 0:
            dG[j] = Xf.T @ (dW_j[:, None] * Xf)
            dB[j] = np.empty((0, p), dtype=np.float64)
            dM[j] = np.empty((0, 0), dtype=np.float64)

    H_full = np.zeros((n_sp, n_sp), dtype=np.float64)
    H_legacy = np.zeros_like(H_full)

    for j in range(n_sp):
        Pj = P_derivs[j]
        group_j = groups.get(j)
        for k in range(j, n_sp):
            Pk = P_derivs[k]
            dbeta_j = dbeta[j]
            dbeta_k = dbeta[k]
            deta_j = deta[j]
            deta_k = deta[k]

            delta_jk = 1.0 if j == k else 0.0
            d2beta_jk = -(
                A_inv
                @ (
                    dA[k] @ dbeta_j
                    + Pj @ dbeta_k
                    + delta_jk * (Pj @ beta)
                )
            )
            d2eta_jk = X @ d2beta_jk
            d2W_jk = d2W_eta * deta_j * deta_k + dW_eta * d2eta_jk
            d2beta_mat[j][k] = d2beta_jk
            d2beta_mat[k][j] = d2beta_jk

            hij = float(dbeta_k @ (Pj @ beta))
            if j == k:
                hij += 0.5 * float(beta @ (Pj @ beta))

            d2XtWX_jk = X.T @ (d2W_jk[:, None] * X)
            d2A_jk = d2XtWX_jk + (float(sp[j]) * Pj if j == k else 0.0)
            d2XtWX_mat[j][k] = d2XtWX_jk
            d2XtWX_mat[k][j] = d2XtWX_jk
            d2A_mat[j][k] = d2A_jk
            d2A_mat[k][j] = d2A_jk

            if q > 0:
                d2M_jk = Zr.T @ (d2W_jk[:, None] * Zr)
                if j == k and group_j is not None and group_j.size > 0:
                    d2M_jk[np.ix_(group_j, group_j)] += float(sp[j]) * np.eye(group_j.size, dtype=np.float64)

                hij += 0.5 * float(
                    np.trace(-Minv @ dM[k] @ Minv @ dM[j] + Minv @ d2M_jk)
                )

                if method == "REML" and p > 0:
                    d2B_jk = Zr.T @ (d2W_jk[:, None] * Xf)
                    d2G_jk = Xf.T @ (d2W_jk[:, None] * Xf)
                    dC_j = dG[j] - dB[j].T @ Minv @ B0 - B0.T @ Minv @ dB[j] + B0.T @ Minv @ dM[j] @ Minv @ B0
                    dC_k = dG[k] - dB[k].T @ Minv @ B0 - B0.T @ Minv @ dB[k] + B0.T @ Minv @ dM[k] @ Minv @ B0

                    d2C_jk = (
                        d2G_jk
                        - d2B_jk.T @ Minv @ B0
                        + dB[j].T @ Minv @ dM[k] @ Minv @ B0
                        - dB[j].T @ Minv @ dB[k]
                        - dB[k].T @ Minv @ dB[j]
                        + B0.T @ Minv @ dM[k] @ Minv @ dB[j]
                        - B0.T @ Minv @ d2B_jk
                        + dB[k].T @ Minv @ dM[j] @ Minv @ B0
                        - B0.T @ Minv @ dM[k] @ Minv @ dM[j] @ Minv @ B0
                        + B0.T @ Minv @ d2M_jk @ Minv @ B0
                        - B0.T @ Minv @ dM[j] @ Minv @ dM[k] @ Minv @ B0
                        + B0.T @ Minv @ dM[j] @ Minv @ dB[k]
                    )
                    hij += 0.5 * float(np.trace(-C_inv @ dC_k @ C_inv @ dC_j + C_inv @ d2C_jk))
            elif method == "REML" and p > 0:
                d2G_jk = Xf.T @ (d2W_jk[:, None] * Xf)
                dC_j = dG[j]
                dC_k = dG[k]
                hij += 0.5 * float(np.trace(-C_inv @ dC_k @ C_inv @ dC_j + C_inv @ d2G_jk))

            H_full[j, k] = hij
            H_full[k, j] = hij
            H_legacy[j, k] = hij
            H_legacy[k, j] = hij

    detXWXS1 = detXWXS2 = detS1 = detS2 = D1 = D2 = None

    try:
        bSb, bSb1_store, bSb2_store = _penalty_quadratic_and_sp_derivatives(
            beta=beta,
            P_total=np.asarray(sol["P"], dtype=np.float64),
            P_derivs=P_derivs,
            dbeta_cols=dbeta,
            d2beta_mat=d2beta_mat,
        )
        dVkk = _quadratic_form_in_beta_directions(np.asarray(sol["A"], dtype=np.float64), dbeta)
        det1, det2 = _logdet_penalized_system_derivatives(
            A_inv=np.asarray(sol["A_inv"], dtype=np.float64),
            dA=dA,
            d2A_mat=d2A_mat,
        )
        trA, trA1, trA2 = _hat_matrix_trace_and_sp_derivatives(
            A_inv=np.asarray(sol["A_inv"], dtype=np.float64),
            XtWX=np.asarray(sol["XtWX"], dtype=np.float64),
            dA=dA,
            d2A_mat=d2A_mat,
            dXtWX=dXtWX,
            d2XtWX_mat=d2XtWX_mat,
        )
        setattr(
            model,
            "_pirls_reml_derivative_kernel_state_",
            {
                "bSb": bSb,
                "bSb1": bSb1_store,
                "bSb2": bSb2_store,
                "dVkk": dVkk,
                "det1": det1,
                "det2": det2,
                "trA": trA,
                "trA1": trA1,
                "trA2": trA2,
                "D1": D1,
                "D2": D2,
                "detXWXS1": detXWXS1,
                "detXWXS2": detXWXS2,
                "detS1": detS1,
                "detS2": detS2,
            },
        )
    except Exception:
        setattr(model, "_pirls_reml_derivative_kernel_state_", None)

    return H_full[np.ix_(free_idx, free_idx)]
