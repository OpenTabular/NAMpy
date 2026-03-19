import numpy as np
from scipy.linalg import cho_solve

from .covariance import build_bayes_and_freq_covariances
from .penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
    coerce_fit_offset,
    stabilized_cholesky_solve,
)


def fit_pirls_core(
    Z,
    y,
    penalty_blocks,
    smoothing_params,
    family,
    fit_intercept=True,
    max_iter=100,
    tol=1e-8,
    max_step_halving=25,
    offset=None,
):
    """
    Penalized IRLS core for one-linear-predictor penalized GLMs.

    Returns a criterion-ready state containing the converged working system.
    """
    y = family.validate_y(y)
    Z = np.asarray(Z, dtype=np.float64)
    offset = coerce_fit_offset(offset, Z.shape[0])

    X = build_full_design(Z, fit_intercept=fit_intercept)
    P_full = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        fit_intercept=fit_intercept,
        n_coef=Z.shape[1],
    )

    mu0 = family.initialize_mu(y)
    eta0 = family.link(mu0)
    rhs0 = eta0 if offset is None else (eta0 - offset)

    A0 = X.T @ X + P_full
    b0 = X.T @ rhs0
    beta, _, _, _ = stabilized_cholesky_solve(A0, b0)

    eta = X @ beta if offset is None else offset + X @ beta
    mu = family.inverse_link(eta)
    dev_old = float(family.deviance(y, mu))
    pdev_old = dev_old + float(beta @ (P_full @ beta))

    converged = False
    failed_step = False
    failure_reason = None
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        eta = X @ beta if offset is None else offset + X @ beta
        mu = family.inverse_link(eta)
        mu_eta = np.clip(family.mu_eta(eta), 1e-12, None)
        var = np.clip(family.variance(mu), 1e-12, None)

        fisher_W = np.clip((mu_eta ** 2) / var, 1e-12, None)
        use_fisher = bool(getattr(family, "canonical_link", False))
        if not use_fisher and hasattr(family, "dvar") and hasattr(family, "d2link"):
            try:
                dvar = np.asarray(family.dvar(mu), dtype=np.float64)
                d2link = np.asarray(family.d2link(mu), dtype=np.float64)
                alpha = 1.0 + (y - mu) * (dvar / var + d2link * mu_eta)
                eps_alpha = np.finfo(np.float64).eps
                small = np.abs(alpha) < eps_alpha
                if np.any(small):
                    alpha = alpha.copy()
                    alpha[small] = np.where(alpha[small] >= 0.0, eps_alpha, -eps_alpha)
                W = fisher_W * alpha
                z = eta + (y - mu) / (mu_eta * alpha)
                if np.any(~np.isfinite(W)) or np.any(W <= 0.0) or np.any(~np.isfinite(z)):
                    # Mirror mgcv's practical behavior: fall back to Fisher scoring
                    # for unstable/indefinite Newton updates.
                    use_fisher = True
            except Exception:
                use_fisher = True

        if use_fisher:
            W = fisher_W
            z = eta + (y - mu) / mu_eta

        W = np.clip(W, 1e-12, None)
        z_work = z if offset is None else (z - offset)

        XtW = X.T * W
        XtWX = XtW @ X
        A = XtWX + P_full
        b = XtW @ z_work

        beta_prop, _, _, _ = stabilized_cholesky_solve(A, b)

        beta_new = beta_prop
        eta_new = X @ beta_new if offset is None else offset + X @ beta_new
        mu_new = family.inverse_link(eta_new)
        dev_new = float(family.deviance(y, mu_new))
        pdev_new = dev_new + float(beta_new @ (P_full @ beta_new))

        div_thresh = 10.0 * (0.1 + abs(pdev_old)) * np.sqrt(np.finfo(np.float64).eps)

        invalid = (not np.isfinite(dev_new)) or (not np.isfinite(pdev_new))
        bad_step = invalid or (pdev_new - pdev_old > div_thresh)

        if bad_step:
            beta_half = beta_prop.copy()
            accepted_halfstep = False
            for _ in range(max_step_halving):
                beta_half = 0.5 * (beta + beta_half)
                eta_half = X @ beta_half if offset is None else offset + X @ beta_half
                mu_half = family.inverse_link(eta_half)
                dev_half = float(family.deviance(y, mu_half))
                pdev_half = dev_half + float(beta_half @ (P_full @ beta_half))
                if np.isfinite(dev_half) and np.isfinite(pdev_half) and pdev_half - pdev_old <= div_thresh:
                    beta_new = beta_half
                    eta_new = eta_half
                    mu_new = mu_half
                    dev_new = dev_half
                    pdev_new = pdev_half
                    accepted_halfstep = True
                    break
            if not accepted_halfstep:
                failed_step = True
                failure_reason = "step_halving_exhausted"
                converged = False
                break

        grad = 2.0 * (A @ beta_new - b)
        beta = beta_new
        eta = eta_new
        mu = mu_new

        scale_ref = 1.0 if family.known_scale is not None else max(abs(dev_new), 1.0)
        if abs(pdev_new - pdev_old) < tol * (abs(scale_ref) + abs(pdev_new)):
            if np.max(np.abs(grad)) <= tol * (abs(scale_ref) + abs(pdev_new)):
                converged = True
                dev_old = dev_new
                pdev_old = pdev_new
                break

        dev_old = dev_new
        pdev_old = pdev_new

    eta = X @ beta if offset is None else offset + X @ beta
    mu = family.inverse_link(eta)
    mu_eta = np.clip(family.mu_eta(eta), 1e-12, None)
    var = np.clip(family.variance(mu), 1e-12, None)

    fisher_W = np.clip((mu_eta ** 2) / var, 1e-12, None)
    use_fisher = bool(getattr(family, "canonical_link", False))
    if not use_fisher and hasattr(family, "dvar") and hasattr(family, "d2link"):
        try:
            dvar = np.asarray(family.dvar(mu), dtype=np.float64)
            d2link = np.asarray(family.d2link(mu), dtype=np.float64)
            alpha = 1.0 + (y - mu) * (dvar / var + d2link * mu_eta)
            eps_alpha = np.finfo(np.float64).eps
            small = np.abs(alpha) < eps_alpha
            if np.any(small):
                alpha = alpha.copy()
                alpha[small] = np.where(alpha[small] >= 0.0, eps_alpha, -eps_alpha)
            W = fisher_W * alpha
            z = eta + (y - mu) / (mu_eta * alpha)
            if np.any(~np.isfinite(W)) or np.any(W <= 0.0) or np.any(~np.isfinite(z)):
                use_fisher = True
        except Exception:
            use_fisher = True
    if use_fisher:
        W = fisher_W
        z = eta + (y - mu) / mu_eta
    W = np.clip(W, 1e-12, None)
    z_work = z if offset is None else (z - offset)

    XtW = X.T * W
    XtWX = XtW @ X
    A = XtWX + P_full

    _, cA, loA, _ = stabilized_cholesky_solve(
        A, np.zeros(X.shape[1], dtype=np.float64)
    )
    A_inv = cho_solve((cA, loA), np.eye(A.shape[0]), check_finite=False)

    H_coef = A_inv @ XtWX
    trace_H = float(np.trace(H_coef))
    edf = trace_H

    scale = float(family.estimate_dispersion(y, mu, edf=edf))
    deviance = float(family.deviance(y, mu))
    rss = float(np.sum((y - mu) ** 2))
    penalty_quadratic = float(beta @ (P_full @ beta))
    loglik = float(family.loglik(y, mu, scale=scale))

    Vp, Vf, H_coef = build_bayes_and_freq_covariances(scale, A_inv, XtWX)

    if fit_intercept:
        intercept = float(beta[0])
        beta_term = beta[1:].copy()
    else:
        intercept = 0.0
        beta_term = beta.copy()

    return {
        "coef_full": beta.copy(),
        "intercept": intercept,
        "beta": beta_term,
        "eta": eta,
        "mu": mu,
        "rss": rss,
        "deviance": deviance,
        "edf": edf,
        "trace_H": trace_H,
        "scale": scale,
        "cov_bayes": Vp,
        "cov_freq": Vf,
        "H_coef": H_coef,
        "X": X,
        "A": A,
        "A_inv": A_inv,
        "XtWX": XtWX,
        "P": P_full,
        "penalty_matrix": P_full,
        "working_weights": W,
        "fisher_weights": fisher_W,
        "working_response": z_work,
        "penalty_quadratic": penalty_quadratic,
        "loglik": loglik,
        "converged": converged,
        "iter": n_iter,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "offset": None if offset is None else offset.copy(),
    }
