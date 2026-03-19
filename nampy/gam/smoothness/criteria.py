import numpy as np
from scipy.linalg import cho_factor, cho_solve


def gcv_score_gaussian(model, y, log_smoothing_params):
    sp = model._expand_smoothing_params_from_log(log_smoothing_params)
    sol = model._solve_gaussian_given_smoothing(y, sp)
    n = model.n_samples_
    den = 1.0 - model.score_gamma * sol["trace_H"] / n
    if den <= 1e-12 or not np.isfinite(den):
        return np.inf
    return (sol["rss"] / n) / (den ** 2)


def criterion_gcv_gaussian(model, y, log_sp):
    return gcv_score_gaussian(model, y, log_sp)


def criterion_ml_reml_exact(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the exact Gaussian ML/REML/LAML path."
        )

    if not model._can_use_exact_gaussian_ml_reml():
        raise NotImplementedError(
            "Exact Gaussian ML/REML/LAML is currently available only for "
            "terms with one or more disjoint-support primary smooth penalties, "
            "plus at most one null-space penalty per support block."
        )

    y = model.family.validate_y(y)
    y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
    sp = model._expand_smoothing_params_from_log(log_sp)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    n = Xf.shape[0]
    p = model.rank_X_fix_
    q = model.n_rand_

    if q == 0:
        if p == 0:
            rss_v = max(float(y_eff @ y_eff), 1e-14)
            return n * np.log(rss_v / n)

        XtX = Xf.T @ Xf
        try:
            cXtX, lo = cho_factor(XtX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        b_hat = cho_solve((cXtX, lo), Xf.T @ y_eff, check_finite=False)
        resid = y_eff - Xf @ b_hat
        rss_v = max(float(resid @ resid), 1e-14)

        if method == "ML":
            return n * np.log(rss_v / n)

        if n <= p:
            return np.inf

        logdet_XtX = 2.0 * float(np.sum(np.log(np.diag(cXtX))))
        return (n - p) * np.log(rss_v / (n - p)) + logdet_XtX

    lam_vec = _laplace_lambda_vector(model, sp)

    if np.any(lam_vec <= 0):
        return np.inf

    M = model.ZtZ_rand_ + np.diag(lam_vec)
    try:
        cM, loM = cho_factor(M, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

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
            return np.inf

        XtKy = Xf.T @ Ky
        b_hat = cho_solve((cXKX, loXKX), XtKy, check_finite=False)
        rss_v = max(float(y_eff @ Ky - XtKy @ b_hat), 1e-14)
    else:
        cXKX = None
        rss_v = max(float(y_eff @ Ky), 1e-14)

    logdet_M = 2.0 * float(np.sum(np.log(np.diag(cM))))
    logdet_Lam = float(np.sum(np.log(lam_vec)))
    logdet_Vtilde = logdet_M - logdet_Lam

    if method == "ML":
        return n * np.log(rss_v / n) + logdet_Vtilde

    if n <= p:
        return np.inf

    logdet_XtKX = 0.0 if p == 0 else 2.0 * float(
        np.sum(np.log(np.abs(np.diag(cXKX))))
    )
    return (n - p) * np.log(rss_v / (n - p)) + logdet_Vtilde + logdet_XtKX


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
    lam_parts = [
        np.full(int(block["n_pen"]), float(sp[int(block["smoothing_index"])]), dtype=np.float64)
        for block in blocks
        if int(block["n_pen"]) > 0
    ]
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


def resolve_ml_reml_scoring_backend(model, method="reml"):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        raise ValueError("method must be one of {'ml', 'reml', 'laml'}")

    if (
        bool(getattr(model.family, "supports_closed_form_solve", False))
        and model._can_use_exact_gaussian_ml_reml()
    ):
        return "gaussian_exact"

    if (
        bool(getattr(model.family, "supports_pirls", False))
        and model._can_use_simple_ml_reml_structure()
    ):
        return "pirls_laplace"

    return None


def criterion_ml_reml(model, y, log_sp, method):
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend == "gaussian_exact":
        return criterion_ml_reml_exact(model, y, log_sp, method.upper())
    if backend == "pirls_laplace":
        branch_method = "REML" if str(method).lower() in {"reml", "laml"} else "ML"
        return criterion_ml_reml_pirls(model, y, log_sp, branch_method)
    model._raise_ml_reml_backend_error(method)
    raise AssertionError("unreachable")


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
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dW_deta = np.asarray(model.family.working_weight_derivative_eta(eta), dtype=np.float64)

    Xf = model.X_fix_
    Zr = model.Z_rand_
    p = int(model.rank_X_fix_)
    q = int(model.n_rand_)
    grad_full = np.zeros(int(model.n_smoothing_params_), dtype=np.float64)

    if q == 0:
        for j, Pj in enumerate(P_derivs):
            if np.any(Pj):
                grad_full[j] = 0.5 * float(beta @ (Pj @ beta))

        if method == "REML" and p > 0:
            XtWX_fix = Xf.T @ (W[:, None] * Xf)
            cFix, loFix = cho_factor(XtWX_fix, check_finite=False)
            C_inv = cho_solve((cFix, loFix), np.eye(p), check_finite=False)
            for j, Pj in enumerate(P_derivs):
                if not np.any(Pj):
                    continue
                dbeta_j = -(A_inv @ (Pj @ beta))
                dW_j = dW_deta * (X @ dbeta_j)
                dXtWX_j = Xf.T @ (dW_j[:, None] * Xf)
                grad_full[j] += 0.5 * float(np.sum(C_inv * dXtWX_j))

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
        deta_j = X @ dbeta_j
        dW_j = dW_deta * deta_j

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
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dW_eta = np.asarray(model.family.working_weight_derivative_eta(eta), dtype=np.float64)
    d2W_eta = np.asarray(
        model.family.working_weight_second_derivative_eta(eta), dtype=np.float64
    )

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

            hij = float(dbeta_k @ (Pj @ beta))
            if j == k:
                hij += 0.5 * float(beta @ (Pj @ beta))

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

    return H_full[np.ix_(free_idx, free_idx)]


def criterion_value(model, y, log_sp, method="gcv"):
    method = str(method).lower()
    if method == "gcv":
        if model._uses_closed_form_solver():
            return criterion_gcv_gaussian(model, y, log_sp)
        return criterion_gcv_pirls(model, y, log_sp)
    if method in {"ubre", "aic", "ubreaic"}:
        return criterion_ubre_pirls(model, y, log_sp)
    if method == "ml":
        return criterion_ml_reml(model, y, log_sp, "ml")
    if method in {"reml", "laml"}:
        return criterion_ml_reml(model, y, log_sp, method)
    raise ValueError(
        "method must be one of "
        "{'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml'}"
    )


def criterion_gradient_numerical(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-5,
    eps_rel=1e-4,
):
    """Centered finite-difference gradient of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty((0,), dtype=np.float64)

    grad = np.empty_like(x)
    f0 = float(criterion_value(model, y, x, method=method))

    if not np.isfinite(f0):
        grad.fill(np.nan)
        return grad

    for i in range(x.size):
        step = max(float(eps_abs), float(eps_rel) * (1.0 + abs(float(x[i]))))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += step
        x_minus[i] -= step

        f_plus = float(criterion_value(model, y, x_plus, method=method))
        f_minus = float(criterion_value(model, y, x_minus, method=method))

        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2.0 * step)
        elif np.isfinite(f_plus):
            grad[i] = (f_plus - f0) / step
        elif np.isfinite(f_minus):
            grad[i] = (f0 - f_minus) / step
        else:
            grad[i] = np.nan

    return grad


def criterion_gradient(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-5,
    eps_rel=1e-4,
):
    method = str(method).lower()
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if backend == "gaussian_exact":
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_exact(model, y, log_sp, exact_method)
        if backend == "pirls_laplace" and bool(
            getattr(model.family, "supports_exact_pirls_first_derivatives", False)
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_pirls_exact(model, y, log_sp, exact_method)

    return criterion_gradient_numerical(
        model,
        y,
        log_sp,
        method=method,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
    )


def criterion_hessian_numerical(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-4,
    eps_rel=1e-3,
):
    """Centered finite-difference Hessian of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    H = np.empty((n, n), dtype=np.float64)
    steps = np.maximum(float(eps_abs), float(eps_rel) * (1.0 + np.abs(x)))

    for j in range(n):
        h = float(steps[j])
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h

        g_plus = criterion_gradient(
            model,
            y,
            x_plus,
            method=method,
            eps_abs=max(eps_abs * 0.1, 1e-6),
            eps_rel=max(eps_rel * 0.1, 1e-5),
        )
        g_minus = criterion_gradient(
            model,
            y,
            x_minus,
            method=method,
            eps_abs=max(eps_abs * 0.1, 1e-6),
            eps_rel=max(eps_rel * 0.1, 1e-5),
        )
        H[:, j] = (g_plus - g_minus) / (2.0 * h)

    return 0.5 * (H + H.T)


def criterion_hessian(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-4,
    eps_rel=1e-3,
):
    method = str(method).lower()
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if (
            backend == "pirls_laplace"
            and method in {"reml", "laml"}
            and bool(
            getattr(model.family, "supports_exact_pirls_second_derivatives", False)
            )
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, exact_method)
    return criterion_hessian_numerical(
        model,
        y,
        log_sp,
        method=method,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
    )


def criterion_infinite_sp_signal(model, y, log_sp, method="reml"):
    method = str(method).lower()
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    backend = None
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)

    exact_method = "REML" if method in {"reml", "laml"} else "ML"

    if (
        backend == "pirls_laplace"
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
        and model._can_use_simple_ml_reml_structure()
    ):
        sp = model._expand_smoothing_params_from_log(x)
        sol = model._solve_pirls_given_smoothing(y, sp)
        beta = np.asarray(sol["coef_full"], dtype=np.float64)
        A = np.asarray(sol["A"], dtype=np.float64)
        A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
        P_derivs = _penalty_derivative_matrices(model, sp)

        grad = np.asarray(
            criterion_gradient_ml_reml_pirls_exact(model, y, x, exact_method),
            dtype=np.float64,
        )
        dvkk = np.zeros(int(model.n_smoothing_params_ or 0), dtype=np.float64)
        for j, Pj in enumerate(P_derivs):
            if not np.any(Pj):
                continue
            dbeta_j = -(A_inv @ (Pj @ beta))
            dvkk[j] = float(dbeta_j @ (A @ dbeta_j))

        free_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_mask = ~free_mask
        return grad, dvkk[free_mask]

    if backend == "gaussian_exact" and model._can_use_exact_gaussian_ml_reml():
        sp = model._expand_smoothing_params_from_log(x)
        sol = model._solve_gaussian_given_smoothing(y, sp)
        beta = np.asarray(sol["coef_full"], dtype=np.float64)
        A = np.asarray(sol["A"], dtype=np.float64)
        A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
        P_derivs = _penalty_derivative_matrices(model, sp)

        grad = np.asarray(
            criterion_gradient_ml_reml_exact(model, y, x, exact_method),
            dtype=np.float64,
        )
        dvkk = np.zeros(int(model.n_smoothing_params_ or 0), dtype=np.float64)
        for j, Pj in enumerate(P_derivs):
            if not np.any(Pj):
                continue
            dbeta_j = -(A_inv @ (Pj @ beta))
            dvkk[j] = float(dbeta_j @ (A @ dbeta_j))

        free_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_mask = ~free_mask
        return grad, dvkk[free_mask]

    grad = np.asarray(criterion_gradient(model, y, x, method=method), dtype=np.float64)
    hess = np.asarray(criterion_hessian(model, y, x, method=method), dtype=np.float64)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1] or hess.shape[0] != n:
        dvkk = np.full(n, np.nan, dtype=np.float64)
    else:
        dvkk = np.diag(hess).astype(np.float64, copy=True)
    return grad, dvkk
