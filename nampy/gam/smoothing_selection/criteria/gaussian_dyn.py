"""Gaussian dynamic REML/LAML joint objective and related designs."""

import numpy as np

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import _coef_column_offset, _n_smoothing_params
from ...fit.model_ops import (
    expand_smoothing_params_from_log,
    solve_gaussian_given_smoothing,
)
from ..reparam import (
    _stable_penalty_logdet_derivatives,
    _static_penalty_null_dim,
    build_penalty_reparameterization_state,
)
from .gaussian_reml_algebra import (
    gaussian_reml_saturation_terms_wrt_variance,
    gaussian_reml_weighted_degrees_and_log_weight_term,
    gaussian_weighted_residual_sum_squares,
    prior_weights_diagonal_from_fit,
    quadratic_form_penalty,
)
from .laplace import _penalty_derivative_matrices
from .pirls_deriv import _gdi1_kernel


def _cache_gaussian_reml_scale_est(model, sol) -> None:
    """Cache the current fixed-fit Gaussian scale estimate for outer-Newton scaling."""
    try:
        scale = float(sol["scale"])
    except Exception:
        scale = np.nan
    model._gaussian_reml_last_scale_est_ = scale


def _gaussian_penalty_quadratic_mgcv_style(model, sol, sp) -> float:
    beta = np.asarray(sol["coef_full"], dtype=np.float64).ravel()
    if beta.size == 0:
        return 0.0
    state = build_penalty_reparameterization_state(
        model,
        np.asarray(sol["X"], dtype=np.float64),
        np.asarray(sp, dtype=np.float64),
        deriv=0,
    )
    alpha = np.linalg.solve(np.asarray(state.T, dtype=np.float64), beta)
    St = np.asarray(state.St, dtype=np.float64)
    return float(alpha @ (St @ alpha))


def _gaussian_dynamic_reml_derivative_terms(model, y, log_sp, method):
    y = model.family.validate_y(y)
    y if model.offset_train_ is None else (y - model.offset_train_)
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_gaussian_given_smoothing(model, y, sp)
    _cache_gaussian_reml_scale_est(model, sol)

    if method not in {"REML", "LAML"}:
        raise NotImplementedError(
            "Exact dynamic Gaussian derivatives are currently implemented for REML/LAML only."
        )

    np.asarray(sol["A"], dtype=np.float64)
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    n_s = int(model.n_samples_)
    w1 = prior_weights_diagonal_from_fit(sol, n_s)
    yv = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(sol["mu"], dtype=np.float64).ravel()
    dev = gaussian_weighted_residual_sum_squares(yv, mu, w1)
    Pq = quadratic_form_penalty(
        np.asarray(sol["coef_full"], dtype=np.float64),
        np.asarray(sol["penalty_matrix"], dtype=np.float64),
    )
    F = float(dev) + float(Pq)
    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    n_weighted = float(nu + Mp)
    prof_df = n_weighted - gamma * Mp
    coeff = prof_df / gamma if gamma > 0.0 else np.nan
    scale = F / prof_df if prof_df > 0.0 else np.nan
    if (
        not np.isfinite(gamma)
        or gamma <= 0.0
        or not np.isfinite(scale)
        or scale <= 0.0
        or not np.isfinite(F)
        or F <= 0.0
        or not np.isfinite(prof_df)
        or prof_df <= 0.0
        or not np.isfinite(coeff)
        or coeff <= 0.0
    ):
        n_sp = int(_n_smoothing_params(model) or 0)
        return {
            "valid": False,
            "grad": np.full(n_sp, np.nan, dtype=np.float64),
            "hess": np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        }

    P_derivs = _penalty_derivative_matrices(model, sp)
    _logdet_S, logdetS_grad, logdetS_hess = _stable_penalty_logdet_derivatives(
        model, sp, order=2
    )
    if not np.all(np.isfinite(logdetS_grad)) or not np.all(np.isfinite(logdetS_hess)):
        n_sp = int(_n_smoothing_params(model) or 0)
        return {
            "valid": False,
            "grad": np.full(n_sp, np.nan, dtype=np.float64),
            "hess": np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        }

    n_sp = int(_n_smoothing_params(model) or 0)
    grad_full = np.zeros(n_sp, dtype=np.float64)
    hess_full = np.zeros((n_sp, n_sp), dtype=np.float64)
    beta1 = [np.zeros_like(beta) for _ in range(n_sp)]
    F1 = np.zeros(n_sp, dtype=np.float64)
    F2 = np.zeros((n_sp, n_sp), dtype=np.float64)
    logdetA1 = np.zeros(n_sp, dtype=np.float64)
    logdetA2 = np.zeros((n_sp, n_sp), dtype=np.float64)
    U = [None] * n_sp

    for j, Pj in enumerate(P_derivs):
        if not np.any(Pj):
            continue
        Uj = A_inv @ Pj
        U[j] = Uj
        beta_j = -(Uj @ beta)
        beta1[j] = beta_j
        F1[j] = float(beta @ (Pj @ beta))
        logdetA1[j] = float(np.trace(Uj))
        grad_full[j] = 0.5 * (coeff * F1[j] / F + logdetA1[j] - logdetS_grad[j])

    for j, Pj in enumerate(P_derivs):
        if not np.any(Pj):
            continue
        for k, Pk in enumerate(P_derivs[j:], start=j):
            if not np.any(Pk):
                continue
            delta = 1.0 if j == k else 0.0
            -(A_inv @ (Pk @ beta1[j] + Pj @ beta1[k])) + delta * beta1[j]
            Fjk = 2.0 * float(beta1[k] @ (Pj @ beta)) + delta * F1[j]
            F2[j, k] = Fjk
            F2[k, j] = Fjk
            logdetAjk = -float(np.trace(U[k] @ U[j])) + delta * logdetA1[j]
            logdetA2[j, k] = logdetAjk
            logdetA2[k, j] = logdetAjk
            val = 0.5 * (
                coeff * (Fjk / F - (F1[j] * F1[k]) / (F * F))
                + logdetAjk
                - logdetS_hess[j, k]
            )
            hess_full[j, k] = val
            hess_full[k, j] = val

    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    return {
        "valid": True,
        "grad": grad_full[free_mask],
        "hess": hess_full[np.ix_(free_mask, free_mask)],
        "F": float(F),
        "coeff": float(coeff),
        "F1_free": np.asarray(F1[free_mask], dtype=np.float64).copy(),
        "F2_free": np.asarray(
            F2[np.ix_(free_mask, free_mask)], dtype=np.float64
        ).copy(),
        "logdetA1_free": np.asarray(logdetA1[free_mask], dtype=np.float64).copy(),
        "logdetS1_free": np.asarray(logdetS_grad[free_mask], dtype=np.float64).copy(),
        "logdetA2_free": np.asarray(
            logdetA2[np.ix_(free_mask, free_mask)], dtype=np.float64
        ).copy(),
        "logdetS2_free": np.asarray(
            logdetS_hess[np.ix_(free_mask, free_mask)], dtype=np.float64
        ).copy(),
    }


def criterion_ml_reml_gaussian_dynamic_joint(
    model, y, log_sp_free, log_sigma2, method="REML"
):
    """
    Gaussian REML/LAML criterion with an explicit error variance sigma^2 = exp(log_sigma2).

    Standard GAM outer optimisation for Gaussian REML/LAML jointly updates log smoothing
    parameters and ``log(sigma^2)``. The profiled criterion elsewhere uses
    ``sigma^2 = F / nu``; this function is the unconcentrated form for that joint loop.

    With prior weights, the doubled score subtracts ``(n_eff/n_row)*sum(log w[w>0])`` and
    uses ``nu = (n_eff/n_row)*sum(w>0) - Mp`` in the ``log(2*pi*sigma^2)`` term. Set
    ``model.n_true_`` for a custom effective row count ``n_eff`` (default ``n_row``).
    """
    method_u = str(method).upper()
    if method_u not in {"REML", "LAML"}:
        raise ValueError("method must be 'REML' or 'LAML' for the joint Gaussian path.")

    y = model.family.validate_y(y)
    sp = expand_smoothing_params_from_log(
        model, np.asarray(log_sp_free, dtype=np.float64).ravel()
    )
    sol = solve_gaussian_given_smoothing(model, y, sp)
    _cache_gaussian_reml_scale_est(model, sol)
    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    nu, sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    n_weighted = float(nu + Mp)
    coeff = n_weighted / gamma - Mp if gamma > 0.0 else np.nan
    if not np.isfinite(gamma) or gamma <= 0.0 or not np.isfinite(coeff) or coeff <= 0.0:
        return np.inf
    yv = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(sol["mu"], dtype=np.float64).ravel()
    dev = gaussian_weighted_residual_sum_squares(yv, mu, w1)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    Pq = float(kernel.bSb)
    rss_bSb = float(dev) + float(Pq)
    if not np.isfinite(rss_bSb) or rss_bSb <= 0.0:
        return np.inf
    sigma2 = float(np.exp(float(log_sigma2)))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return np.inf
    ldet_xtwx_plus_penalty = kernel.ldet_XWX_plus_S
    if ldet_xtwx_plus_penalty is None or not np.isfinite(float(ldet_xtwx_plus_penalty)):
        return np.inf
    logdet_S = _stable_penalty_logdet_derivatives(model, sp, order=0)[0]
    if not np.isfinite(logdet_S):
        return np.inf
    if not np.isfinite(sum_log_scaled):
        return np.inf
    ls = gaussian_reml_saturation_terms_wrt_variance(w1, sigma2)
    if np.any(~np.isfinite(ls)):
        return np.inf
    fac = 1.0
    if np.isfinite(nobs) and nobs > 0.0 and n_eff is not None and np.isfinite(n_eff):
        fac = float(n_eff) / nobs
    ls0 = fac * float(ls[0])
    return (
        (rss_bSb / (2.0 * sigma2) - ls0) / gamma
        + float(ldet_xtwx_plus_penalty) / 2.0
        - float(logdet_S) / 2.0
        - (Mp / 2.0) * (np.log(2.0 * np.pi * sigma2) - np.log(gamma))
    )


def criterion_ml_reml_gaussian_dynamic_profiled(model, y, log_sp_free, method="REML"):
    """
    Gaussian REML/LAML criterion profiled over sigma^2 using the same joint
    objective as `criterion_ml_reml_gaussian_dynamic_joint`.

    This matches mgcv's outer-optimization geometry: the reported profiled score
    is the joint objective evaluated at the analytic optimum
    `sigma^2 = (dev + beta'Sbeta) / nu`.
    """
    method_u = str(method).upper()
    if method_u not in {"REML", "LAML"}:
        raise ValueError(
            "method must be 'REML' or 'LAML' for the profiled Gaussian path."
        )

    y = model.family.validate_y(y)
    sp = expand_smoothing_params_from_log(
        model, np.asarray(log_sp_free, dtype=np.float64).ravel()
    )
    sol = solve_gaussian_given_smoothing(model, y, sp)
    _cache_gaussian_reml_scale_est(model, sol)

    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    prof_df = float(nu + Mp - gamma * Mp)
    if (
        not np.isfinite(gamma)
        or gamma <= 0.0
        or not np.isfinite(prof_df)
        or prof_df <= 0.0
    ):
        return np.inf

    yv = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(sol["mu"], dtype=np.float64).ravel()
    dev = gaussian_weighted_residual_sum_squares(yv, mu, w1)
    Pq = quadratic_form_penalty(
        np.asarray(sol["coef_full"], dtype=np.float64),
        np.asarray(sol["penalty_matrix"], dtype=np.float64),
    )
    F = float(dev) + float(Pq)
    if not np.isfinite(F) or F <= 0.0:
        return np.inf

    log_sigma2 = float(np.log(max(F / prof_df, LOG_GUARD_MIN)))
    return criterion_ml_reml_gaussian_dynamic_joint(
        model,
        y,
        log_sp_free,
        log_sigma2,
        method=method_u,
    )


def criterion_gradient_ml_reml_gaussian_dynamic_joint(
    model, y, log_sp_free, log_sigma2, method="REML"
):
    """
    Gradient w.r.t. (log(sp_free...), log(sigma^2)) for `criterion_ml_reml_gaussian_dynamic_joint`.

    Uses the profiled REML derivatives from `_gaussian_dynamic_reml_derivative_terms` and
    corrects the smoothing-parameter block for a fixed sigma^2 (not profiled).
    """
    method_u = str(method).upper()
    y = model.family.validate_y(y)
    sp = expand_smoothing_params_from_log(
        model, np.asarray(log_sp_free, dtype=np.float64).ravel()
    )
    sol = solve_gaussian_given_smoothing(model, y, sp)
    _cache_gaussian_reml_scale_est(model, sol)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    F = float(sol["deviance"]) + float(kernel.bSb)
    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    n_weighted = float(nu + Mp)
    coeff = n_weighted / gamma - Mp if gamma > 0.0 else np.nan
    if not np.isfinite(F) or abs(F) < LOG_GUARD_MIN:
        return None
    sigma2 = float(np.exp(float(log_sigma2)))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return None
    if not np.isfinite(gamma) or gamma <= 0.0:
        return None
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    Dp1 = np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64)
    K1 = np.asarray(kernel.K1, dtype=np.float64)
    g_sp = Dp1[free_mask] / (2.0 * gamma * sigma2) + K1[free_mask]
    g_tau = 0.5 * (coeff - F / (gamma * sigma2))
    return np.concatenate([g_sp, np.array([g_tau], dtype=np.float64)])


def criterion_hessian_ml_reml_gaussian_dynamic_joint(
    model, y, log_sp_free, log_sigma2, method="REML"
):
    """
    Hessian of the joint Gaussian REML/LAML criterion with respect to
    ``(log(sp_free...), log(sigma^2))``.

    The ``(log(sp_free...))`` block is the profiled Hessian from
    ``_gaussian_dynamic_reml_derivative_terms`` plus the joint variance term
    ``F/(gamma*sigma^2)``. Off-diagonal terms are
    ``-0.5 * dF/dsp_i / (gamma*sigma^2)`` and the variance block is
    ``0.5 * F / (gamma*sigma^2)``.
    """
    method_u = str(method).upper()
    y = model.family.validate_y(y)
    sp = expand_smoothing_params_from_log(
        model, np.asarray(log_sp_free, dtype=np.float64).ravel()
    )
    sol = solve_gaussian_given_smoothing(model, y, sp)
    _cache_gaussian_reml_scale_est(model, sol)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return None

    F = float(sol["deviance"]) + float(kernel.bSb)
    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    n_weighted = float(nu + Mp)
    coeff = n_weighted / gamma - Mp if gamma > 0.0 else np.nan
    if not np.isfinite(F) or F <= 0.0:
        return None
    if not np.isfinite(coeff) or coeff <= 0.0:
        return None

    sigma2 = float(np.exp(float(log_sigma2)))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return None

    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    Dp1 = np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64)
    Dp2 = np.asarray(kernel.D2 + kernel.bSb2, dtype=np.float64)
    K2 = np.asarray(kernel.K2, dtype=np.float64)
    h_sp = (
        Dp2[np.ix_(free_mask, free_mask)] / (2.0 * gamma * sigma2)
        + K2[np.ix_(free_mask, free_mask)]
    )

    n_free = int(h_sp.shape[0])
    h = np.zeros((n_free + 1, n_free + 1), dtype=np.float64)
    h[:n_free, :n_free] = h_sp
    h_cross = -Dp1[free_mask] / (2.0 * gamma * sigma2)
    h[:n_free, n_free] = h_cross
    h[n_free, :n_free] = h_cross
    h[n_free, n_free] = 0.5 * F / (gamma * sigma2)
    return h
