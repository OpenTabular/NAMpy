"""Exact first/second derivatives of PIRLS Laplace ML/REML criteria."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma
from .pirls_reml_derivative_blocks import (
    _deviance_chained_to_smoothing,
    _deviance_coefficient_derivatives,
    _hat_matrix_trace_and_sp_derivatives,
    _logdet_penalized_system_derivatives,
    _pearson_coefficient_derivatives,
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
from .penalty import _stable_penalty_logdet_derivatives, _static_penalty_null_dim
from .pirls import _solve_gamma_profile_scale, _gamma_profile_objective_curvature


def _gamma_saturated_loglik_scale_derivatives(y, scale):
    y = np.asarray(y, dtype=np.float64)
    n = float(y.size)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.nan, np.nan
    inv_scale = 1.0 / scale
    l1 = float(n * (digamma(inv_scale) + np.log(scale)) / (scale ** 2))
    l2 = float(
        n
        * (
            -polygamma(1, inv_scale) / scale
            + (1.0 - 2.0 * np.log(scale) - 2.0 * digamma(inv_scale))
        )
        / (scale ** 3)
    )
    return l1, l2


def _gamma_joint_kernel_state(model, y, log_sp, method):
    method = str(method).upper()
    _ = criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, method)
    state = getattr(model, "_pirls_reml_gamma_state_", None)
    if not isinstance(state, dict):
        raise RuntimeError("Gamma joint PIRLS derivatives require exact fixed-sp gamma kernel state.")
    mp = float(
        _static_penalty_null_dim(model)
        + int(bool(getattr(model, "fit_intercept", False)))
    )
    return state, mp


def criterion_gradient_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError("Joint PIRLS Gamma derivatives are implemented only for family='gamma'.")

    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method)
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool))) if model.smoothing_fixed_mask_ is not None else int(model.n_smoothing_params_ or 0)
        return np.full(n_free + 1, np.nan, dtype=np.float64)

    _, score_lphi, _ = _gamma_profile_objective_curvature(
        model,
        y,
        float(state["Dp"]),
        phi,
        mp,
        method=str(method).upper(),
    )
    free_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    grad_sp = np.asarray(state["Dp1"], dtype=np.float64) / (2.0 * phi) + np.asarray(
        state["K1"], dtype=np.float64
    )
    return np.concatenate(
        [
            np.asarray(grad_sp[free_mask], dtype=np.float64),
            np.array([float(score_lphi)], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError("Joint PIRLS Gamma derivatives are implemented only for family='gamma'.")

    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method)
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool))) if model.smoothing_fixed_mask_ is not None else int(model.n_smoothing_params_ or 0)
        return np.full((n_free + 1, n_free + 1), np.nan, dtype=np.float64)

    _, _, curv_lphi = _gamma_profile_objective_curvature(
        model,
        y,
        float(state["Dp"]),
        phi,
        mp,
        method=str(method).upper(),
    )
    free_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask

    H_sp = np.asarray(state["Dp2"], dtype=np.float64) / (2.0 * phi) + np.asarray(
        state["K2"], dtype=np.float64
    )
    cross = -np.asarray(state["Dp1"], dtype=np.float64) / (2.0 * phi)
    H_free = np.asarray(H_sp[np.ix_(free_mask, free_mask)], dtype=np.float64)
    cross_free = np.asarray(cross[free_mask], dtype=np.float64)
    out = np.zeros((int(np.sum(free_mask)) + 1, int(np.sum(free_mask)) + 1), dtype=np.float64)
    out[:-1, :-1] = H_free
    out[:-1, -1] = cross_free
    out[-1, :-1] = cross_free
    out[-1, -1] = float(curv_lphi)
    return out


def _pirls_tensor_coefficient_space_term_and_gradient(model, sol, sp, dA):
    A = np.asarray(sol["A"], dtype=np.float64)
    try:
        cA, _ = cho_factor(A, check_finite=False)
    except np.linalg.LinAlgError:
        n_sp = len(dA)
        return np.nan, np.full(n_sp, np.nan, dtype=np.float64)

    logdet_A = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
    logdet_S, detS1, _ = _stable_penalty_logdet_derivatives(model, sp, order=1)
    if not np.isfinite(logdet_S):
        n_sp = len(dA)
        return np.inf, np.full(n_sp, np.nan, dtype=np.float64)

    detA1, _ = _logdet_penalized_system_derivatives(
        A_inv=np.asarray(sol["A_inv"], dtype=np.float64),
        dA=dA,
        d2A_mat=[[np.zeros_like(A, dtype=np.float64) for _ in dA] for _ in dA],
    )
    return 0.5 * (logdet_A - logdet_S), 0.5 * (detA1 - detS1)


def _pirls_tensor_coefficient_space_term_derivatives(model, sol, sp, dA, d2A_mat):
    A = np.asarray(sol["A"], dtype=np.float64)
    try:
        cA, _ = cho_factor(A, check_finite=False)
    except np.linalg.LinAlgError:
        n_sp = len(dA)
        return (
            np.nan,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )

    logdet_A = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
    logdet_S, detS1, detS2 = _stable_penalty_logdet_derivatives(model, sp, order=2)
    if not np.isfinite(logdet_S):
        n_sp = len(dA)
        return (
            np.inf,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )

    detA1, detA2 = _logdet_penalized_system_derivatives(
        A_inv=np.asarray(sol["A_inv"], dtype=np.float64),
        dA=dA,
        d2A_mat=d2A_mat,
    )
    return (
        0.5 * (logdet_A - logdet_S),
        0.5 * (detA1 - detS1),
        0.5 * (detA2 - detS2),
    )


def _pirls_laplace_term_and_gradient(model, sol, sp, dbeta_cols, dW_deta, *, method):
    X = np.asarray(sol["X"], dtype=np.float64)
    Xf = model.X_fix_
    Zr = model.Z_rand_
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    n_sp = int(model.n_smoothing_params_ or 0)

    K = 0.0
    K1 = np.zeros(n_sp, dtype=np.float64)

    if q == 0:
        if method == "ML" or p == 0:
            return K, K1

        XtWX_fix = Xf.T @ (W[:, None] * Xf)
        cFix, loFix = cho_factor(XtWX_fix, check_finite=False)
        C_inv = cho_solve((cFix, loFix), np.eye(p), check_finite=False)
        K = 0.5 * (2.0 * float(np.sum(np.log(np.abs(np.diag(cFix))))))
        for j, dbeta_j in enumerate(dbeta_cols):
            deta_j = X @ np.asarray(dbeta_j, dtype=np.float64)
            dW_j = dW_deta * deta_j
            dXtWX_j = Xf.T @ (dW_j[:, None] * Xf)
            K1[j] = 0.5 * float(np.sum(C_inv * dXtWX_j))
        return K, K1

    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.nan, np.full(n_sp, np.nan, dtype=np.float64)

    groups = _lambda_group_indices(model)
    ZtW = Zr.T * W[np.newaxis, :]
    B = ZtW @ Xf if p > 0 else np.empty((q, 0), dtype=np.float64)
    M = ZtW @ Zr + np.diag(lam_vec)
    cM, loM = cho_factor(M, check_finite=False)
    Minv = cho_solve((cM, loM), np.eye(q), check_finite=False)

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    logdet_Lam = float(np.sum(np.log(lam_vec)))
    K = 0.5 * (logdet_M - logdet_Lam)

    if method == "REML" and p > 0:
        XtKX = Xf.T @ (W[:, None] * Xf) - B.T @ Minv @ B
        cC, loC = cho_factor(XtKX, check_finite=False)
        C_inv = cho_solve((cC, loC), np.eye(p), check_finite=False)
        K += 0.5 * (2.0 * float(np.sum(np.log(np.abs(np.diag(cC))))))
    else:
        C_inv = None

    for j, dbeta_j in enumerate(dbeta_cols):
        deta_j = X @ np.asarray(dbeta_j, dtype=np.float64)
        dW_j = dW_deta * deta_j
        dM_j = Zr.T @ (dW_j[:, None] * Zr)
        group = groups.get(j)
        if group is not None and group.size > 0:
            dM_j[np.ix_(group, group)] += float(sp[j]) * np.eye(group.size, dtype=np.float64)

        grad_j = 0.5 * float(np.sum(Minv * dM_j))
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

        K1[j] = grad_j

    return K, K1


def _pirls_laplace_term_derivatives(
    model,
    sol,
    sp,
    dbeta_cols,
    d2beta_mat,
    dW_eta,
    d2W_eta,
    *,
    method,
):
    X = np.asarray(sol["X"], dtype=np.float64)
    Xf = model.X_fix_
    Zr = model.Z_rand_
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    n_sp = int(model.n_smoothing_params_ or 0)

    K = 0.0
    K1 = np.zeros(n_sp, dtype=np.float64)
    K2 = np.zeros((n_sp, n_sp), dtype=np.float64)

    deta = [X @ np.asarray(db, dtype=np.float64) for db in dbeta_cols]
    dW = [np.asarray(dW_eta, dtype=np.float64) * dj for dj in deta]

    if q == 0:
        if method == "ML" or p == 0:
            return K, K1, K2

        C = Xf.T @ (W[:, None] * Xf)
        cC, loC = cho_factor(C, check_finite=False)
        C_inv = cho_solve((cC, loC), np.eye(p), check_finite=False)
        K = 0.5 * (2.0 * float(np.sum(np.log(np.abs(np.diag(cC))))))

        dC = [None] * n_sp
        d2C = [[None] * n_sp for _ in range(n_sp)]
        for j in range(n_sp):
            dC_j = Xf.T @ (dW[j][:, None] * Xf)
            dC[j] = dC_j
            K1[j] = 0.5 * float(np.trace(C_inv @ dC_j))

        for j in range(n_sp):
            for k in range(j, n_sp):
                d2eta_jk = X @ np.asarray(d2beta_mat[j][k], dtype=np.float64)
                d2W_jk = np.asarray(d2W_eta, dtype=np.float64) * deta[j] * deta[k] + np.asarray(
                    dW_eta, dtype=np.float64
                ) * d2eta_jk
                d2C_jk = Xf.T @ (d2W_jk[:, None] * Xf)
                d2C[j][k] = d2C_jk
                d2C[k][j] = d2C_jk
                val = 0.5 * float(np.trace(C_inv @ d2C_jk - C_inv @ dC[k] @ C_inv @ dC[j]))
                K2[j, k] = val
                K2[k, j] = val
        return K, K1, K2

    groups = _lambda_group_indices(model)
    lam_vec = _laplace_lambda_vector(model, sp)
    if np.any(lam_vec <= 0):
        return np.nan, np.full(n_sp, np.nan), np.full((n_sp, n_sp), np.nan)

    ZtW = Zr.T * W[np.newaxis, :]
    B0 = ZtW @ Xf if p > 0 else np.empty((q, 0), dtype=np.float64)
    G0 = Xf.T @ (W[:, None] * Xf) if p > 0 else np.empty((0, 0), dtype=np.float64)
    M = ZtW @ Zr + np.diag(lam_vec)
    cM, loM = cho_factor(M, check_finite=False)
    Minv = cho_solve((cM, loM), np.eye(q), check_finite=False)
    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    logdet_Lam = float(np.sum(np.log(lam_vec)))
    K = 0.5 * (logdet_M - logdet_Lam)

    dM = [None] * n_sp
    dB = [None] * n_sp
    dG = [None] * n_sp
    dC = [None] * n_sp
    d2M = [[None] * n_sp for _ in range(n_sp)]
    d2B = [[None] * n_sp for _ in range(n_sp)]
    d2G = [[None] * n_sp for _ in range(n_sp)]
    d2C = [[None] * n_sp for _ in range(n_sp)]

    for j in range(n_sp):
        dM_j = Zr.T @ (dW[j][:, None] * Zr)
        group_j = groups.get(j)
        if group_j is not None and group_j.size > 0:
            dM_j[np.ix_(group_j, group_j)] += float(sp[j]) * np.eye(group_j.size, dtype=np.float64)
        dM[j] = dM_j
        K1[j] = 0.5 * float(np.trace(Minv @ dM_j))
        if group_j is not None and group_j.size > 0:
            K1[j] -= 0.5 * float(group_j.size)
        if p > 0:
            dB_j = Zr.T @ (dW[j][:, None] * Xf)
            dG_j = Xf.T @ (dW[j][:, None] * Xf)
            dB[j] = dB_j
            dG[j] = dG_j
        else:
            dB[j] = np.empty((q, 0), dtype=np.float64)
            dG[j] = np.empty((0, 0), dtype=np.float64)

    if method == "REML" and p > 0:
        C = G0 - B0.T @ Minv @ B0
        cC, loC = cho_factor(C, check_finite=False)
        C_inv = cho_solve((cC, loC), np.eye(p), check_finite=False)
        K += 0.5 * (2.0 * float(np.sum(np.log(np.abs(np.diag(cC))))))
        for j in range(n_sp):
            dC_j = (
                dG[j]
                - dB[j].T @ Minv @ B0
                - B0.T @ Minv @ dB[j]
                + B0.T @ Minv @ dM[j] @ Minv @ B0
            )
            dC[j] = dC_j
            K1[j] += 0.5 * float(np.trace(C_inv @ dC_j))
    else:
        C = C_inv = None

    for j in range(n_sp):
        for k in range(j, n_sp):
            d2eta_jk = X @ np.asarray(d2beta_mat[j][k], dtype=np.float64)
            d2W_jk = np.asarray(d2W_eta, dtype=np.float64) * deta[j] * deta[k] + np.asarray(
                dW_eta, dtype=np.float64
            ) * d2eta_jk
            group_j = groups.get(j)

            d2M_jk = Zr.T @ (d2W_jk[:, None] * Zr)
            if j == k and group_j is not None and group_j.size > 0:
                d2M_jk[np.ix_(group_j, group_j)] += float(sp[j]) * np.eye(group_j.size, dtype=np.float64)
            d2M[j][k] = d2M_jk
            d2M[k][j] = d2M_jk

            val = 0.5 * float(np.trace(Minv @ d2M_jk - Minv @ dM[k] @ Minv @ dM[j]))
            if method == "REML" and p > 0:
                d2B_jk = Zr.T @ (d2W_jk[:, None] * Xf)
                d2G_jk = Xf.T @ (d2W_jk[:, None] * Xf)
                d2B[j][k] = d2B_jk
                d2B[k][j] = d2B_jk
                d2G[j][k] = d2G_jk
                d2G[k][j] = d2G_jk
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
                d2C[j][k] = d2C_jk
                d2C[k][j] = d2C_jk
                val += 0.5 * float(np.trace(C_inv @ d2C_jk - C_inv @ dC[k] @ C_inv @ dC[j]))

            K2[j, k] = val
            K2[k, j] = val

    return K, K1, K2



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
    family_name = str(getattr(model.family, "name", "")).lower()
    if getattr(model.family, "known_scale", None) is None and family_name != "gamma":
        raise NotImplementedError(
            "Exact PIRLS ML/REML gradients are currently implemented only for "
            "fixed-scale families, plus Gamma via the profiled scale branch."
        )
    if getattr(model.family, "known_scale", None) is None and family_name == "gamma":
        free_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_mask = ~free_mask
        if int(np.sum(free_mask)) == 0:
            return np.empty((0,), dtype=np.float64)

        _ensure_penalty_reparameterization(model)
        sp = model._expand_smoothing_params_from_log(log_sp)
        sol = model._solve_pirls_given_smoothing(y, sp)

        X = np.asarray(sol["X"], dtype=np.float64)
        beta = np.asarray(sol["coef_full"], dtype=np.float64)
        eta = np.asarray(sol["eta"], dtype=np.float64)
        W = np.asarray(sol["working_weights"], dtype=np.float64)
        A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
        P_derivs = _penalty_derivative_matrices(model, sp)

        n_sp = int(model.n_smoothing_params_ or 0)
        dbeta = [None] * n_sp
        zero_d2 = [[None] * n_sp for _ in range(n_sp)]
        for j, Pj in enumerate(P_derivs):
            dbj = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
            dbeta[j] = dbj
            for k in range(n_sp):
                zero_d2[j][k] = np.zeros_like(beta)

        dev_grad, dev_hess = _deviance_coefficient_derivatives(
            model,
            y,
            eta,
            sol["mu"],
            W,
            X,
        )
        D1, _ = _deviance_chained_to_smoothing(dev_grad, dev_hess, dbeta, zero_d2)
        bSb, bSb1, _ = _penalty_quadratic_and_sp_derivatives(
            beta=beta,
            P_total=np.asarray(sol["P"], dtype=np.float64),
            P_derivs=P_derivs,
            dbeta_cols=dbeta,
            d2beta_mat=zero_d2,
        )
        mp = float(
            _static_penalty_null_dim(model)
            + int(bool(getattr(model, "fit_intercept", False)))
        )
        Dp = float(sol["deviance"]) + float(bSb)
        phi = _solve_gamma_profile_scale(
            model,
            y,
            Dp,
            mp,
            method=method,
            init_scale=float(sol["scale"]),
        )
        if not np.isfinite(phi) or phi <= 0.0:
            raise RuntimeError("Gamma exact PIRLS gradient requires positive profile scale.")
        Dp1 = np.asarray(D1 + bSb1, dtype=np.float64)

        dW_eta = np.asarray(
            _working_weight_derivatives_wrt_linpred(model, y, eta, sol["mu"], W)[0],
            dtype=np.float64,
        )
        dA = [None] * n_sp
        for j, Pj in enumerate(P_derivs):
            deta_j = X @ np.asarray(dbeta[j], dtype=np.float64)
            dW_j = dW_eta * deta_j
            dA[j] = X.T @ (dW_j[:, None] * X) + np.asarray(Pj, dtype=np.float64)

        if model._has_tensor_terms():
            K, K1 = _pirls_tensor_coefficient_space_term_and_gradient(model, sol, sp, dA)
        else:
            K, K1 = _pirls_laplace_term_and_gradient(
                model,
                sol,
                sp,
                dbeta,
                dW_eta,
                method=method,
            )

        Dp1_free = Dp1[free_mask]
        K1_free = K1[free_mask]
        setattr(
            model,
            "_pirls_reml_gamma_state_",
            {
                "K": float(K),
                "K1": np.asarray(K1, dtype=np.float64),
                "phi": float(phi),
                "scale_est": float(sol["scale"]),
                "Dp": float(Dp),
                "Dp1": np.asarray(Dp1, dtype=np.float64),
            },
        )
        return Dp1_free / (2.0 * phi) + K1_free

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

    if model._has_tensor_terms():
        dev_grad, dev_hess = _deviance_coefficient_derivatives(
            model,
            y,
            eta,
            sol["mu"],
            W,
            X,
        )
        zero_d2 = [[None] * int(model.n_smoothing_params_ or 0) for _ in range(int(model.n_smoothing_params_ or 0))]
        dA = [None] * int(model.n_smoothing_params_ or 0)
        dbeta_store = [np.zeros_like(beta, dtype=np.float64) for _ in range(int(model.n_smoothing_params_ or 0))]
        for j, Pj in enumerate(P_derivs):
            dbeta_j = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
            dbeta_store[j] = dbeta_j
            deta_j = X @ dbeta_j
            dW_j = dW_deta * deta_j
            dA[j] = X.T @ (dW_j[:, None] * X) + np.asarray(Pj, dtype=np.float64)
            for k in range(int(model.n_smoothing_params_ or 0)):
                zero_d2[j][k] = np.zeros_like(beta)
        D1, _ = _deviance_chained_to_smoothing(dev_grad, dev_hess, dbeta_store, zero_d2)
        _, bSb1, _ = _penalty_quadratic_and_sp_derivatives(
            beta=beta,
            P_total=np.asarray(sol["P"], dtype=np.float64),
            P_derivs=P_derivs,
            dbeta_cols=dbeta_store,
            d2beta_mat=zero_d2,
        )
        _, K1 = _pirls_tensor_coefficient_space_term_and_gradient(model, sol, sp, dA)
        scale = float(sol["scale"])
        grad_full = np.asarray(D1 + bSb1, dtype=np.float64) / (2.0 * scale) + np.asarray(K1, dtype=np.float64)
        return grad_full[free_mask]

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    grad_full = np.zeros(int(model.n_smoothing_params_), dtype=np.float64)
    dA_store = [None] * int(model.n_smoothing_params_)

    dbeta_store = [np.zeros_like(beta, dtype=np.float64) for _ in range(int(model.n_smoothing_params_ or 0))]

    if q == 0:
        for j, Pj in enumerate(P_derivs):
            if np.any(Pj):
                grad_full[j] = 0.5 * float(beta @ (Pj @ beta))
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
    family_name = str(getattr(model.family, "name", "")).lower()
    if getattr(model.family, "known_scale", None) is None and family_name != "gamma":
        raise NotImplementedError(
            "Exact PIRLS ML/REML Hessians are currently implemented only for fixed-scale families, "
            "plus Gamma via the profiled scale branch."
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
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dW_eta, d2W_eta = _working_weight_derivatives_wrt_linpred(model, y, eta, sol["mu"], W)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)

    n_sp = int(model.n_smoothing_params_ or 0)
    grad_full = np.zeros(n_sp, dtype=np.float64)
    penalty_grad_raw = np.zeros(n_sp, dtype=np.float64)
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

        if not np.any(Pj):
            continue
        penalty_grad_raw[j] = 0.5 * float(beta @ (Pj @ beta))
        grad_j = float(penalty_grad_raw[j])
        if q > 0:
            group_j = groups.get(j)
            grad_j += 0.5 * float(np.sum(Minv * dM[j]))
            if group_j is not None and group_j.size > 0:
                grad_j -= 0.5 * float(group_j.size)
            if method == "REML" and p > 0:
                dC_j = (
                    dG[j]
                    - dB[j].T @ Minv @ B0
                    - B0.T @ Minv @ dB[j]
                    + B0.T @ Minv @ dM[j] @ Minv @ B0
                )
                grad_j += 0.5 * float(np.sum(C_inv * dC_j))
        elif method == "REML" and p > 0:
            grad_j += 0.5 * float(np.sum(C_inv * dG[j]))
        grad_full[j] = grad_j

    H_full = np.zeros((n_sp, n_sp), dtype=np.float64)
    penalty_hess_raw = np.zeros((n_sp, n_sp), dtype=np.float64)

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
            penalty_hess_raw[j, k] = hij
            penalty_hess_raw[k, j] = hij

            d2XtWX_jk = X.T @ (d2W_jk[:, None] * X)
            d2A_jk = d2XtWX_jk + (Pj if j == k else 0.0)
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

    detXWXS1 = detXWXS2 = None
    D1 = D2 = None
    P1 = P2 = phi1 = phi2 = None
    full_grad = full_hess = None

    try:
        dev_grad, dev_hess = _deviance_coefficient_derivatives(
            model,
            y,
            eta,
            sol["mu"],
            W,
            X,
        )
        D1, D2 = _deviance_chained_to_smoothing(dev_grad, dev_hess, dbeta, d2beta_mat)
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
        if getattr(model.family, "known_scale", None) is None:
            mp = float(
                _static_penalty_null_dim(model)
                + int(bool(getattr(model, "fit_intercept", False)))
            )
            Dp = float(sol["deviance"]) + float(bSb)
            phi = _solve_gamma_profile_scale(
                model,
                y,
                Dp,
                mp,
                method=method,
                init_scale=float(sol["scale"]),
            )
            if not np.isfinite(phi) or phi <= 0.0:
                raise RuntimeError("Gamma exact PIRLS derivatives require positive profile scale.")
            Dp1 = np.asarray(D1 + bSb1_store, dtype=np.float64)
            Dp2 = np.asarray(D2 + bSb2_store, dtype=np.float64)
            _, _, phi_curv = _gamma_profile_objective_curvature(
                model, y, Dp, phi, mp, method=method
            )
            if not np.isfinite(phi_curv) or abs(phi_curv) <= 1e-14:
                raise RuntimeError("Gamma exact PIRLS derivatives require finite profile curvature.")
            if model._has_tensor_terms():
                K, det_grad, det_hess = _pirls_tensor_coefficient_space_term_derivatives(
                    model,
                    sol,
                    sp,
                    dA,
                    d2A_mat,
                )
            else:
                K, det_grad, det_hess = _pirls_laplace_term_derivatives(
                    model,
                    sol,
                    sp,
                    dbeta,
                    d2beta_mat,
                    dW_eta,
                    d2W_eta,
                    method=method,
                )
            joint_grad = det_grad + Dp1 / (2.0 * phi)
            cross = -Dp1 / (2.0 * phi)
            full_grad = joint_grad
            full_hess = det_hess + Dp2 / (2.0 * phi) - np.outer(cross, cross) / phi_curv
            setattr(
                model,
                "_pirls_reml_gamma_state_",
                {
                    "K": float(K),
                    "K1": np.asarray(det_grad, dtype=np.float64),
                    "K2": np.asarray(det_hess, dtype=np.float64),
                    "phi": float(phi),
                    "scale_est": float(sol["scale"]),
                    "phi_curv": float(phi_curv),
                    "Dp": float(Dp),
                    "Dp1": np.asarray(Dp1, dtype=np.float64),
                    "Dp2": np.asarray(Dp2, dtype=np.float64),
                },
            )
        elif model._has_tensor_terms():
            scale = float(sol["scale"])
            _, det_grad, det_hess = _pirls_tensor_coefficient_space_term_derivatives(
                model,
                sol,
                sp,
                dA,
                d2A_mat,
            )
            full_grad = (np.asarray(D1, dtype=np.float64) + np.asarray(bSb1_store, dtype=np.float64)) / (
                2.0 * scale
            ) + det_grad
            full_hess = (np.asarray(D2, dtype=np.float64) + np.asarray(bSb2_store, dtype=np.float64)) / (
                2.0 * scale
            ) + det_hess
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
                "P1": P1,
                "P2": P2,
                "phi1": phi1,
                "phi2": phi2,
                "full_grad": full_grad,
                "full_hess": full_hess,
                "penalty_grad_raw": penalty_grad_raw,
                "penalty_hess_raw": penalty_hess_raw,
                "detXWXS1": detXWXS1,
                "detXWXS2": detXWXS2,
            },
        )
    except Exception:
        setattr(model, "_pirls_reml_derivative_kernel_state_", None)

    out_hess = full_hess if full_hess is not None else H_full
    return out_hess[np.ix_(free_idx, free_idx)]
