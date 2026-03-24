"""
Algebra building blocks for exact first- and second-order derivatives of REML/ML
smoothness-selection criteria with respect to log-smoothing parameters.

These functions compute the implicit-function derivatives of the fitted coefficient
vector and working system matrices through the penalized IRLS fixed point, then
combine them into the score derivatives needed by the outer optimiser.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr


def _working_weight_derivatives_wrt_linpred(model, y, eta, mu, w):
    family = model.family
    fisher = bool(getattr(family, "canonical_link", False))
    if fisher:
        d1 = np.asarray(family.working_weight_derivative_eta(eta, y=y), dtype=np.float64)
        d2 = np.asarray(family.working_weight_second_derivative_eta(eta, y=y), dtype=np.float64)
        return d1, d2

    g1 = 1.0 / np.clip(np.asarray(family.mu_eta(eta), dtype=np.float64), 1e-14, None)
    V = np.clip(np.asarray(family.variance(mu), dtype=np.float64), 1e-14, None)
    V1 = np.asarray(family.dvar(mu), dtype=np.float64) / V
    V2 = np.asarray(family.d2var(mu), dtype=np.float64) / V
    V3 = np.asarray(family.d3var(mu), dtype=np.float64) / V
    g2 = np.asarray(family.d2link(mu), dtype=np.float64) / g1
    g3 = np.asarray(family.d3link(mu), dtype=np.float64) / g1
    g4 = np.asarray(family.d4link(mu), dtype=np.float64) / g1

    c = np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    alpha = 1.0 + c * (V1 + g2)
    eps_alpha = np.finfo(np.float64).eps
    alpha = alpha.copy()
    alpha[alpha == 0.0] = eps_alpha

    xx = V2 - V1 * V1 + g3 - g2 * g2
    alpha1 = (-(V1 + g2) + c * xx) / alpha
    alpha2 = (
        -2.0 * xx
        + c * (V3 - 3.0 * V1 * V2 + 2.0 * V1 * V1 * V1 + g4 - 3.0 * g3 * g2 + 2.0 * g2 * g2 * g2)
    ) / alpha

    w = np.asarray(w, dtype=np.float64)
    a1 = w * (alpha1 - V1 - 2.0 * g2) / g1
    w_safe = np.clip(w, 1e-14, None)
    a2 = (
        a1 * (a1 / w_safe - g2 / g1)
        - w * (alpha1 * alpha1 - alpha2 + V2 - V1 * V1 + 2.0 * g3 - 2.0 * g2 * g2) / (g1 * g1)
    )
    return a1, a2


def _penalty_quadratic_and_sp_derivatives(beta, P_total, P_derivs, dbeta_cols, d2beta_mat):
    beta = np.asarray(beta, dtype=np.float64)
    P_total = np.asarray(P_total, dtype=np.float64)
    M = len(P_derivs)
    Sb = P_total @ beta
    Skb = [np.asarray(Pj, dtype=np.float64) @ beta for Pj in P_derivs]
    bSb = float(beta @ Sb)
    bSb1 = np.zeros(M, dtype=np.float64)
    bSb2 = np.zeros((M, M), dtype=np.float64)
    for j in range(M):
        dbj = np.asarray(dbeta_cols[j], dtype=np.float64)
        bSb1[j] = float(beta @ Skb[j] + 2.0 * (dbj @ Sb))
    for j in range(M):
        dbj = np.asarray(dbeta_cols[j], dtype=np.float64)
        for k in range(j, M):
            dbk = np.asarray(dbeta_cols[k], dtype=np.float64)
            d2b = np.asarray(d2beta_mat[j][k], dtype=np.float64)
            val = float(
                2.0 * (d2b @ Sb)
                + 2.0 * (dbk @ (P_total @ dbj))
                + 2.0 * (dbj @ Skb[k])
                + 2.0 * (dbk @ Skb[j])
            )
            if j == k:
                val += bSb1[j]
            bSb2[j, k] = val
            bSb2[k, j] = val
    return bSb, bSb1, bSb2


def _quadratic_form_in_beta_directions(A, dbeta_cols):
    A = np.asarray(A, dtype=np.float64)
    if len(dbeta_cols) == 0:
        return np.empty((0, 0), dtype=np.float64)
    B = np.column_stack([np.asarray(v, dtype=np.float64) for v in dbeta_cols])
    return B.T @ A @ B


def _logdet_penalized_system_derivatives(A_inv, dA, d2A_mat):
    A_inv = np.asarray(A_inv, dtype=np.float64)
    M = len(dA)
    det1 = np.zeros(M, dtype=np.float64)
    det2 = np.zeros((M, M), dtype=np.float64)
    for j in range(M):
        dAj = np.asarray(dA[j], dtype=np.float64)
        det1[j] = float(np.trace(A_inv @ dAj))
    for j in range(M):
        dAj = np.asarray(dA[j], dtype=np.float64)
        for k in range(j, M):
            dAk = np.asarray(dA[k], dtype=np.float64)
            d2Ajk = np.asarray(d2A_mat[j][k], dtype=np.float64)
            val = float(np.trace(A_inv @ d2Ajk - A_inv @ dAj @ A_inv @ dAk))
            det2[j, k] = val
            det2[k, j] = val
    return det1, det2


def _hat_matrix_trace_and_sp_derivatives(A_inv, XtWX, dA, d2A_mat, dXtWX, d2XtWX_mat):
    A_inv = np.asarray(A_inv, dtype=np.float64)
    XtWX = np.asarray(XtWX, dtype=np.float64)
    H = A_inv @ XtWX
    M = len(dA)
    trA = float(np.trace(H))
    dH = [None] * M
    trA1 = np.zeros(M, dtype=np.float64)
    trA2 = np.zeros((M, M), dtype=np.float64)
    for j in range(M):
        dAj = np.asarray(dA[j], dtype=np.float64)
        dXtj = np.asarray(dXtWX[j], dtype=np.float64)
        dHj = -A_inv @ dAj @ H + A_inv @ dXtj
        dH[j] = dHj
        trA1[j] = float(np.trace(dHj))
    for j in range(M):
        dAj = np.asarray(dA[j], dtype=np.float64)
        dHj = dH[j]
        for k in range(j, M):
            dAk = np.asarray(dA[k], dtype=np.float64)
            d2Ajk = np.asarray(d2A_mat[j][k], dtype=np.float64)
            dXtj = np.asarray(dXtWX[j], dtype=np.float64)
            d2Xtjk = np.asarray(d2XtWX_mat[j][k], dtype=np.float64)
            dAinvk = -A_inv @ dAk @ A_inv
            dHk = dH[k]
            d2Hjk = (
                -dAinvk @ dAj @ H
                - A_inv @ d2Ajk @ H
                - A_inv @ dAj @ dHk
                + dAinvk @ dXtj
                + A_inv @ d2Xtjk
            )
            val = float(np.trace(d2Hjk))
            trA2[j, k] = val
            trA2[k, j] = val
    return trA, trA1, trA2


def _penalty_rank_scaling_derivatives(model):
    M = int(model.n_smoothing_params_ or 0)
    detS1 = np.zeros(M, dtype=np.float64)
    detS2 = np.zeros((M, M), dtype=np.float64)
    blocks = getattr(model, "_reparam_rand_blocks_", None)
    if not blocks:
        return detS1, detS2
    for block in blocks:
        j = int(block.get("smoothing_index", -1))
        n_pen = int(block.get("n_pen", 0))
        if 0 <= j < M and n_pen > 0:
            detS1[j] += float(n_pen)
    return detS1, detS2


def _deviance_coefficient_derivatives(model, y, eta, mu, W, X):
    family = model.family
    g1 = np.clip(np.asarray(family.mu_eta(eta), dtype=np.float64), 1e-14, None)
    V = np.clip(np.asarray(family.variance(mu), dtype=np.float64), 1e-14, None)
    resid = np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    v1 = -2.0 * resid / (V * g1)
    dev_grad = np.asarray(X, dtype=np.float64).T @ v1
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    good = np.isfinite(W) & (W != 0.0)
    if not np.any(good):
        dev_hess = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        return dev_grad, dev_hess

    Xg = X[good, :]
    wg = W[good]
    WX = np.sqrt(np.abs(wg))[:, None] * Xg

    Q, R, piv = qr(WX, mode="economic", pivoting=True, check_finite=False)
    R1 = np.zeros_like(R, dtype=np.float64)
    for j, pj in enumerate(np.asarray(piv, dtype=np.int64)):
        R1[:, pj] = R[:, j]
    dev_hess = R1.T @ R1

    neg_idx = np.flatnonzero(wg < 0.0)
    if neg_idx.size > 0:
        IQ = Q[neg_idx, :]
        if IQ.size > 0:
            _, s, Vt = np.linalg.svd(IQ, full_matrices=False)
            V = Vt.T
            VD2Vt = (V * (s**2)[np.newaxis, :]) @ V.T
            Kcorr = R1.T @ VD2Vt @ R1
            dev_hess -= 2.0 * Kcorr

    dev_hess = 2.0 * dev_hess
    return dev_grad, dev_hess


def _deviance_chained_to_smoothing(dev_grad, dev_hess, dbeta_cols, d2beta_mat):
    M = len(dbeta_cols)
    D1 = np.zeros(M, dtype=np.float64)
    D2 = np.zeros((M, M), dtype=np.float64)
    dev_grad = np.asarray(dev_grad, dtype=np.float64)
    dev_hess = np.asarray(dev_hess, dtype=np.float64)
    for j in range(M):
        dbj = np.asarray(dbeta_cols[j], dtype=np.float64)
        D1[j] = float(dbj @ dev_grad)
    for j in range(M):
        dbj = np.asarray(dbeta_cols[j], dtype=np.float64)
        for k in range(j, M):
            dbk = np.asarray(dbeta_cols[k], dtype=np.float64)
            d2b = np.asarray(d2beta_mat[j][k], dtype=np.float64)
            val = float(dbk @ dev_hess @ dbj) + float(dev_grad @ d2b)
            D2[j, k] = val
            D2[k, j] = val
    return D1, D2
