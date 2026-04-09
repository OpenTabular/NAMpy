"""
Penalized IRLS core loop for non-Gaussian one-predictor GAMs.

Minimises the penalized deviance ``D(y, mu) + beta' S beta`` where ``S`` is
the total penalty at fixed smoothing parameters.  The fitted mean satisfies
the GLM link ``g(mu) = eta = X beta + offset``.

Algorithm
---------
Each iteration:
1. Compute working weights ``W`` and adjusted response ``z`` from current ``mu``.
   Fisher scoring is used for canonical-link families; Newton weights are
   attempted for non-canonical links and fall back to Fisher if unstable.
2. Solve the penalized normal equations ``(X'WX + S) beta = X'Wz`` via
   stabilised Cholesky.
3. Step-halve if the new penalized deviance diverges or is non-finite.
4. Check convergence.

Returns all matrices needed by downstream smoothness-selection criteria
(hat matrix trace, covariance matrices, working system).
"""

import numpy as np
from scipy.linalg import cho_solve

from ..covariance import build_bayes_and_freq_covariances
from ..penalized_system import (
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
    weights=None,
    coef_start=None,
    etastart=None,
    mustart=None,
):
    """
    Penalized IRLS for one-linear-predictor penalized GLMs.

    Parameters
    ----------
    Z
        Term design matrix (without intercept column), shape ``(n, p)``.
    y
        Response vector, shape ``(n,)``.
    penalty_blocks
        List of penalty block objects (each with ``.matrix``, ``.coef_slice``,
        ``.smoothing_index``).
    smoothing_params
        Linear-scale smoothing parameters, shape ``(K,)``.
    family
        A :class:`nampy.gam.families.base.GLMFamily` instance.
    fit_intercept
        Whether to prepend an intercept column to ``Z``.
    max_iter
        Maximum IRLS iterations.
    tol
        Convergence tolerance for penalized deviance.
    max_step_halving
        Maximum number of step halvings per iteration.
    offset
        Fixed additive offset on the link scale, shape ``(n,)``.
    weights
        Optional prior observation weights, shape ``(n,)``. These scale the
        PIRLS working weights and therefore the penalized normal equations.
    coef_start
        Optional warm-start coefficient vector.

    Returns
    -------
    dict
        Converged working system: ``coef_full``, ``eta``, ``mu``, ``deviance``,
        ``edf``, ``scale``, covariance matrices ``cov_bayes``/``cov_freq``,
        working matrices ``X``, ``A``, ``A_inv``, ``XtWX``, and convergence flags.
    """
    y = family.validate_y(y)
    Z = np.asarray(Z, dtype=np.float64)
    offset = coerce_fit_offset(offset, Z.shape[0])
    if weights is None:
        weights = np.ones(Z.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape != (Z.shape[0],):
        raise ValueError("weights must have length nrow(Z).")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative.")

    X = build_full_design(Z, fit_intercept=fit_intercept)
    P_full = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        fit_intercept=fit_intercept,
        n_coef=Z.shape[1],
    )

    def _working_response_terms(eta_curr, mu_curr):
        mu_eta_curr = np.asarray(family.mu_eta(eta_curr), dtype=np.float64)
        var_curr = np.asarray(family.variance(mu_curr), dtype=np.float64)
        good_curr = (
            (weights > 0.0)
            & np.isfinite(mu_eta_curr)
            & (mu_eta_curr != 0.0)
            & np.isfinite(var_curr)
            & (var_curr > 0.0)
        )
        if not np.any(good_curr):
            return None

        y_g_curr = y[good_curr]
        mu_g_curr = mu_curr[good_curr]
        eta_g_curr = eta_curr[good_curr]
        mu_eta_g_curr = mu_eta_curr[good_curr]
        var_g_curr = var_curr[good_curr]
        weights_g_curr = weights[good_curr]
        off_g_curr = None if offset is None else offset[good_curr]

        fisher_W_curr = weights_g_curr * (mu_eta_g_curr**2) / var_g_curr
        use_fisher_curr = bool(getattr(family, "canonical_link", False))
        if (
            not use_fisher_curr
            and hasattr(family, "dvar")
            and hasattr(family, "d2link")
        ):
            try:
                dvar_curr = np.asarray(family.dvar(mu_g_curr), dtype=np.float64)
                d2link_curr = np.asarray(family.d2link(mu_g_curr), dtype=np.float64)
                alpha_curr = 1.0 + (y_g_curr - mu_g_curr) * (
                    dvar_curr / var_g_curr + d2link_curr * mu_eta_g_curr
                )
                eps_alpha_curr = np.finfo(np.float64).eps
                zero_curr = alpha_curr == 0.0
                if np.any(zero_curr):
                    alpha_curr = alpha_curr.copy()
                    alpha_curr[zero_curr] = eps_alpha_curr
                w_curr = fisher_W_curr * alpha_curr
                z_curr = (
                    eta_g_curr if off_g_curr is None else (eta_g_curr - off_g_curr)
                ) + (y_g_curr - mu_g_curr) / (mu_eta_g_curr * alpha_curr)
                if np.any(~np.isfinite(w_curr)) or np.any(~np.isfinite(z_curr)):
                    use_fisher_curr = True
            except Exception:
                use_fisher_curr = True
        if use_fisher_curr:
            w_curr = fisher_W_curr
            z_curr = (
                eta_g_curr if off_g_curr is None else (eta_g_curr - off_g_curr)
            ) + (y_g_curr - mu_g_curr) / mu_eta_g_curr

        return {
            "good": good_curr,
            "w": np.asarray(w_curr, dtype=np.float64),
            "z": np.asarray(z_curr, dtype=np.float64),
        }

    beta = None
    if coef_start is not None:
        beta0 = np.asarray(coef_start, dtype=np.float64).ravel()
        if beta0.shape == (X.shape[1],) and np.all(np.isfinite(beta0)):
            beta = beta0.copy()

    if beta is None:
        # Default start path: begin from family link(mustart),
        # not from a pre-solved penalized normal equation.
        beta = np.zeros(X.shape[1], dtype=np.float64)

    if etastart is not None:
        eta = np.asarray(etastart, dtype=np.float64).ravel()
        if eta.shape != (X.shape[0],):
            raise ValueError("etastart must have length nrow(X).")
        mu = family.inverse_link(eta)
    elif mustart is not None:
        mu0 = np.asarray(mustart, dtype=np.float64).ravel()
        if mu0.shape != (X.shape[0],):
            raise ValueError("mustart must have length nrow(X).")
        mu = mu0
        eta = family.link(mu)
    else:
        eta = (
            family.link(family.initialize_mu(y))
            if coef_start is None
            else (X @ beta if offset is None else offset + X @ beta)
        )
        mu = family.inverse_link(eta)
    null_beta = np.zeros_like(beta)
    null_eta = X @ null_beta if offset is None else offset + X @ null_beta
    null_mu = family.inverse_link(null_eta)
    old_pdev = float(family.deviance(y, null_mu) + null_beta @ (P_full @ null_beta))

    # If link(mustart) is immediately invalid, shrink toward null eta.
    # Initialization safeguard for invalid starting mu.
    for _ in range(20):
        if np.all(np.isfinite(mu)) and np.all(np.isfinite(eta)):
            break
        eta = 0.9 * eta + 0.1 * null_eta
        mu = family.inverse_link(eta)

    if coef_start is not None:
        old_pdev = float(family.deviance(y, mu) + beta @ (P_full @ beta))

    mu = family.inverse_link(eta)
    float(family.deviance(y, mu))
    pdev_old = float(old_pdev)

    converged = False
    failed_step = False
    failure_reason = None
    n_iter = 0
    inner_trace = []

    for it in range(max_iter):
        n_iter = it + 1
        mu = family.inverse_link(eta)
        mu_eta = np.asarray(family.mu_eta(eta), dtype=np.float64)
        var = np.asarray(family.variance(mu), dtype=np.float64)
        good = (
            (weights > 0.0)
            & np.isfinite(mu_eta)
            & (mu_eta != 0.0)
            & np.isfinite(var)
            & (var > 0.0)
        )
        if not np.any(good):
            failed_step = True
            failure_reason = "no_informative_observations"
            converged = False
            break

        mu_eta_g = mu_eta[good]
        var_g = var[good]
        y_g = y[good]
        mu_g = mu[good]
        eta_g = eta[good]
        X_g = X[good, :]
        weights_g = weights[good]
        off_g = None if offset is None else offset[good]

        fisher_W = weights_g * (mu_eta_g**2) / var_g
        use_fisher = bool(getattr(family, "canonical_link", False))
        if not use_fisher and hasattr(family, "dvar") and hasattr(family, "d2link"):
            try:
                dvar = np.asarray(family.dvar(mu_g), dtype=np.float64)
                d2link = np.asarray(family.d2link(mu_g), dtype=np.float64)
                alpha = 1.0 + (y_g - mu_g) * (dvar / var_g + d2link * mu_eta_g)
                eps_alpha = np.finfo(np.float64).eps
                zero = alpha == 0.0
                if np.any(zero):
                    alpha = alpha.copy()
                    alpha[zero] = eps_alpha
                W = fisher_W * alpha
                z = (eta_g if off_g is None else (eta_g - off_g)) + (y_g - mu_g) / (
                    mu_eta_g * alpha
                )
                if np.any(~np.isfinite(W)) or np.any(~np.isfinite(z)):
                    # Newton weights are unstable; fall back to Fisher scoring.
                    use_fisher = True
            except Exception:
                use_fisher = True

        if use_fisher:
            W = fisher_W
            z = (eta_g if off_g is None else (eta_g - off_g)) + (y_g - mu_g) / mu_eta_g

        eta_lin = eta_g if off_g is None else (eta_g - off_g)
        wz = W * eta_lin + weights_g * mu_eta_g * (y_g - mu_g) / var_g
        good_z = np.isfinite(z) & np.isfinite(W)
        if np.any(~good_z):
            z_work = np.asarray(z, dtype=np.float64).copy()
            bad_z = ~np.isfinite(z_work)
            if np.any(bad_z):
                z_work[bad_z] = 0.0
            good_wz = np.isfinite(W) & np.isfinite(wz)
            X_g = X_g[good_wz, :]
            W = W[good_wz]
            z_work = np.asarray(wz[good_wz], dtype=np.float64)
        else:
            z_work = z

        XtW = X_g.T * W
        XtWX = XtW @ X_g
        A = XtWX + P_full
        b = XtW @ z_work

        try:
            beta_prop, _, _, _ = stabilized_cholesky_solve(A, b)
        except np.linalg.LinAlgError:
            if use_fisher:
                failed_step = True
                failure_reason = "linear_solve_failed"
                converged = False
                break

            # Mirror mgcv gam.fit4.r lines 394-413 more closely: when the
            # exact Newton-weight system is indefinite, retry the same PIRLS
            # step with only the positive exact weights retained, rather than
            # switching straight to Fisher scoring.
            W_pos = np.where(np.isfinite(W) & (W > 0.0), W, 0.0)
            eta_lin = eta_g if off_g is None else (eta_g - off_g)
            wz_pos_full = W_pos * eta_lin + weights_g * mu_eta_g * (y_g - mu_g) / var_g
            z_pos_full = np.asarray(z, dtype=np.float64).copy()
            bad_z = ~np.isfinite(z_pos_full)
            if np.any(bad_z):
                z_pos_full[bad_z] = 0.0
            good_pos = np.isfinite(z_pos_full) & np.isfinite(W_pos)
            if np.any(~good_pos):
                good_pos = np.isfinite(W_pos) & np.isfinite(wz_pos_full)
                z_pos = np.asarray(wz_pos_full[good_pos], dtype=np.float64)
            else:
                z_pos = np.asarray(z_pos_full[good_pos], dtype=np.float64)
            X_pos = X_g[good_pos, :]
            W_pos = W_pos[good_pos]
            XtW = X_pos.T * W_pos
            XtWX = XtW @ X_pos
            A = XtWX + P_full
            b = XtW @ z_pos
            try:
                beta_prop, _, _, _ = stabilized_cholesky_solve(A, b)
                X_g = X_pos
                W = W_pos
                z = z_pos
                z_work = z_pos
            except np.linalg.LinAlgError:
                failed_step = True
                failure_reason = "linear_solve_failed"
                converged = False
                break

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
            for _ in range(max(max_step_halving, 100)):
                beta_half = 0.5 * (beta + beta_half)
                eta_half = X @ beta_half if offset is None else offset + X @ beta_half
                mu_half = family.inverse_link(eta_half)
                dev_half = float(family.deviance(y, mu_half))
                pdev_half = dev_half + float(beta_half @ (P_full @ beta_half))
                if (
                    np.isfinite(dev_half)
                    and np.isfinite(pdev_half)
                    and pdev_half - pdev_old <= div_thresh
                ):
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

        penalty_new = float(beta_new @ (P_full @ beta_new))
        pdev_step = dev_new + penalty_new
        beta = beta_new
        eta = eta_new
        mu = mu_new

        # EFS (Embedded Fisher Scoring): update theta once per IRLS step when
        # the family has a free dispersion parameter.  Mirrors mgcv gam.fit4.r
        # lines 507-515 (scoreType=="EFS" block).
        theta_efs_enabled = bool(getattr(family, "estimate_theta", False)) and not bool(
            getattr(family, "_disable_theta_efs", False)
        )
        if theta_efs_enabled and hasattr(family, "estimate_theta_mle"):
            theta_efs = family.estimate_theta_mle(y, mu, weights=weights)
            if np.isfinite(theta_efs) and theta_efs > 0.0:
                family.theta = theta_efs
                dev_new = float(family.deviance(y, mu))
        pdev_new = dev_new + penalty_new

        grad_info = _working_response_terms(eta, mu)
        if grad_info is None:
            failed_step = True
            failure_reason = "no_informative_observations"
            converged = False
            break
        good_grad = np.asarray(grad_info["good"], dtype=bool)
        w_grad = np.asarray(grad_info["w"], dtype=np.float64)
        z_grad = np.asarray(grad_info["z"], dtype=np.float64)
        eta_grad = eta[good_grad] if offset is None else (eta - offset)[good_grad]
        X_good = X[good_grad, :]
        grad = 2.0 * (X_good.T @ (w_grad * (eta_grad - z_grad))) + 2.0 * (P_full @ beta)

        scale_ref = 1.0 if family.known_scale is not None else max(abs(dev_new), 1.0)
        pdev_conv = pdev_step if theta_efs_enabled else pdev_new
        if abs(pdev_conv - pdev_old) < tol * (abs(scale_ref) + abs(pdev_conv)):
            inner_trace.append(
                {
                    "iter": int(n_iter),
                    "log_theta": (
                        None
                        if not hasattr(family, "theta")
                        else float(np.log(max(float(family.theta), 1e-300)))
                    ),
                    "deviance": float(dev_new),
                    "penalized_deviance": float(pdev_new),
                    "penalized_deviance_conv": float(pdev_conv),
                    "grad_inf_norm": float(np.max(np.abs(grad))),
                    "converged_here": bool(
                        np.max(np.abs(grad)) <= tol * (abs(scale_ref) + abs(pdev_new))
                    ),
                }
            )
            if np.max(np.abs(grad)) <= tol * (abs(scale_ref) + abs(pdev_new)):
                converged = True
                pdev_old = pdev_new
                break

        inner_trace.append(
            {
                "iter": int(n_iter),
                "log_theta": (
                    None
                    if not hasattr(family, "theta")
                    else float(np.log(max(float(family.theta), 1e-300)))
                ),
                "deviance": float(dev_new),
                "penalized_deviance": float(pdev_new),
                "penalized_deviance_conv": float(pdev_conv),
                "grad_inf_norm": float(np.max(np.abs(grad))),
                "converged_here": False,
            }
        )
        pdev_old = pdev_new

    eta = X @ beta if offset is None else offset + X @ beta
    mu = family.inverse_link(eta)
    mu_eta = np.asarray(family.mu_eta(eta), dtype=np.float64)
    var = np.asarray(family.variance(mu), dtype=np.float64)
    good = (
        (weights > 0.0)
        & np.isfinite(mu_eta)
        & (mu_eta != 0.0)
        & np.isfinite(var)
        & (var > 0.0)
    )
    if not np.any(good):
        raise RuntimeError("No informative observations at PIRLS solution.")
    mu_eta_g = mu_eta[good]
    var_g = var[good]
    y_g = y[good]
    mu_g = mu[good]
    eta_g = eta[good]
    X_g = X[good, :]
    weights_g = weights[good]
    off_g = None if offset is None else offset[good]

    fisher_W_g = weights_g * (mu_eta_g**2) / var_g
    use_fisher = bool(getattr(family, "canonical_link", False))
    if not use_fisher and hasattr(family, "dvar") and hasattr(family, "d2link"):
        try:
            dvar = np.asarray(family.dvar(mu_g), dtype=np.float64)
            d2link = np.asarray(family.d2link(mu_g), dtype=np.float64)
            alpha = 1.0 + (y_g - mu_g) * (dvar / var_g + d2link * mu_eta_g)
            eps_alpha = np.finfo(np.float64).eps
            zero = alpha == 0.0
            if np.any(zero):
                alpha = alpha.copy()
                alpha[zero] = eps_alpha
            W_g = fisher_W_g * alpha
            z_g = (eta_g if off_g is None else (eta_g - off_g)) + (y_g - mu_g) / (
                mu_eta_g * alpha
            )
            if np.any(~np.isfinite(W_g)) or np.any(~np.isfinite(z_g)):
                use_fisher = True
        except Exception:
            use_fisher = True
    if use_fisher:
        W_g = fisher_W_g
        z_g = (eta_g if off_g is None else (eta_g - off_g)) + (y_g - mu_g) / mu_eta_g

    W = np.zeros_like(y, dtype=np.float64)
    W[good] = W_g
    fisher_W = np.zeros_like(y, dtype=np.float64)
    fisher_W[good] = fisher_W_g
    z_work = np.zeros_like(y, dtype=np.float64)
    z_work[good] = z_g

    XtW = X_g.T * W_g
    XtWX = XtW @ X_g
    A = XtWX + P_full

    _, cA, loA, _ = stabilized_cholesky_solve(A, np.zeros(X.shape[1], dtype=np.float64))
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
        "inner_trace": inner_trace,
    }
