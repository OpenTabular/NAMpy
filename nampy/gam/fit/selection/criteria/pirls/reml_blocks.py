"""
Algebra building blocks for exact first- and second-order derivatives of REML/ML
smoothing-parameter selection criteria with respect to log-smoothing parameters.

These functions compute the implicit-function derivatives of the fitted coefficient
vector and working system matrices through the penalized IRLS fixed point, then
combine them into the score derivatives needed by the outer optimiser.
"""

import numpy as np


def _working_weight_derivatives_wrt_linpred(model, y, eta, mu, w):
    family = model.family
    fisher = bool(getattr(family, "canonical_link", False))
    eta = np.asarray(eta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if fisher:
        g1 = 1.0 / np.asarray(family.mu_eta(eta), dtype=np.float64)
        V = np.asarray(family.variance(mu), dtype=np.float64)
        V1 = np.asarray(family.dvar(mu), dtype=np.float64) / V
        V2 = np.asarray(family.d2var(mu), dtype=np.float64) / V
        g2 = np.asarray(family.d2link(mu), dtype=np.float64) / g1
        g3 = np.asarray(family.d3link(mu), dtype=np.float64) / g1
        with np.errstate(divide="ignore", invalid="ignore"):
            a1 = -w * (V1 + 2.0 * g2) / g1
            a2 = a1 * (a1 / w - g2 / g1) - w * (
                V2 - V1 * V1 + 2.0 * g3 - 2.0 * g2 * g2
            ) / (g1 * g1)
        return a1, a2

    g1 = 1.0 / np.asarray(family.mu_eta(eta), dtype=np.float64)
    V = np.asarray(family.variance(mu), dtype=np.float64)
    V1 = np.asarray(family.dvar(mu), dtype=np.float64) / V
    V2 = np.asarray(family.d2var(mu), dtype=np.float64) / V
    V3 = np.asarray(family.d3var(mu), dtype=np.float64) / V
    g2 = np.asarray(family.d2link(mu), dtype=np.float64) / g1
    g3 = np.asarray(family.d3link(mu), dtype=np.float64) / g1
    g4 = np.asarray(family.d4link(mu), dtype=np.float64) / g1

    c = np.asarray(y, dtype=np.float64) - mu
    alpha = 1.0 + c * (V1 + g2)
    eps_alpha = np.finfo(np.float64).eps
    alpha = alpha.copy()
    alpha[alpha == 0.0] = eps_alpha

    xx = V2 - V1 * V1 + g3 - g2 * g2
    alpha1 = (-(V1 + g2) + c * xx) / alpha
    alpha2 = (
        -2.0 * xx
        + c
        * (
            V3
            - 3.0 * V1 * V2
            + 2.0 * V1 * V1 * V1
            + g4
            - 3.0 * g3 * g2
            + 2.0 * g2 * g2 * g2
        )
    ) / alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = w * (alpha1 - V1 - 2.0 * g2) / g1
        a2 = a1 * (a1 / w - g2 / g1) - w * (
            alpha1 * alpha1 - alpha2 + V2 - V1 * V1 + 2.0 * g3 - 2.0 * g2 * g2
        ) / (g1 * g1)
    return a1, a2


def _penalty_quadratic_and_sp_derivatives(
    beta, P_total, P_derivs, dbeta_cols, d2beta_mat
):
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
                val += float(beta @ Skb[j])
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


def _deviance_chained_to_smoothing(dev_grad, dev_hess, dbeta_cols, d2beta_mat):
    """Mirror ``gdi.c::gdi1``'s deviance derivative accumulation.

    In particular, upstream forms the quadratic Hessian block with the
    BLAS-free ``mat.c::getXtMX`` loop before adding ``dev_grad' d2beta``.
    Keeping that accumulation order matters when the penalized deviance is the
    difference of nearly cancelling terms at a REML boundary.
    """
    M = len(dbeta_cols)
    D1 = np.zeros(M, dtype=np.float64)
    D2 = np.zeros((M, M), dtype=np.float64)
    if M == 0:
        return D1, D2
    dev_grad = np.asarray(dev_grad, dtype=np.float64)
    dev_hess = np.asarray(dev_hess, dtype=np.float64)
    dbeta = np.asfortranarray(
        np.column_stack([np.asarray(col, dtype=np.float64) for col in dbeta_cols])
    )
    for j in range(M):
        D1[j] = float(np.dot(dbeta[:, j], dev_grad))

    # Operation-for-operation port of mgcv/src/mat.c::getXtMX(). `work`
    # receives M %*% dbeta[, i], one source coefficient at a time, and the
    # lower triangle is accumulated in row order before being mirrored.
    rank = int(dbeta.shape[0])
    if rank == 0:
        return D1, D2
    work = np.empty(rank, dtype=np.float64)
    for i in range(M):
        for row in range(rank):
            value = dbeta[0, i] * dev_hess[row, 0]
            for col in range(1, rank):
                value += dbeta[col, i] * dev_hess[row, col]
            work[row] = value
        for j in range(i + 1):
            value = 0.0
            for row in range(rank):
                value += work[row] * dbeta[row, j]
            D2[i, j] = value
            D2[j, i] = value

    # gdi.c::gdi1() then advances through packed d2beta columns and adds the
    # coefficient-gradient contraction to the already formed X'MX block.
    for m in range(M):
        for k in range(m, M):
            d2b = np.asarray(d2beta_mat[m][k], dtype=np.float64)
            value = 0.0
            for row in range(rank):
                value += dev_grad[row] * d2b[row]
            D2[k, m] += value
            D2[m, k] = D2[k, m]
    return D1, D2
