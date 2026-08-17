"""Gaussian dynamic REML/LAML joint objective and related designs."""

import numpy as np

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import _coef_column_offset, _n_smoothing_params
from ...fit.backends import solve_gaussian_given_smoothing
from ...fit.smoothing_params import expand_smoothing_params_from_log
from ..reparam import (
    _static_penalty_null_dim,
    build_penalty_reparameterization_state,
)
from .gaussian_reml_algebra import (
    gaussian_reml_saturation_terms_wrt_variance,
    gaussian_reml_weighted_degrees_and_log_weight_term,
    prior_weights_diagonal_from_fit,
)
from .pirls.derivatives import _gdi1_kernel


def _cache_gaussian_reml_scale_est(model, sol) -> None:
    """Cache the current fixed-fit Gaussian scale estimate for outer-Newton scaling."""
    model._gaussian_reml_last_scale_est_ = float(sol["scale"])


def _gaussian_dynamic_deviance(
    sol,
    y: np.ndarray,
    prior_weights: np.ndarray,
) -> float:
    """
    Mirror `mgcv::gam.fit3()` exact-Gaussian REML bookkeeping.

    For Gaussian `gam.fit3` fits, the final `gdi1()` overwrite can change the
    reported coefficients / fitted values while the outer REML score continues to
    use the PIRLS-step deviance stored on the fit object. That state is required
    for a supported Gaussian dynamic objective.
    """
    dev = None
    if isinstance(sol, dict):
        dev = sol.get("deviance", None)
    else:
        dev = getattr(sol, "deviance", None)
    if dev is None or not np.isfinite(float(dev)):
        raise RuntimeError(
            "Gaussian dynamic ML/REML requires the gam.fit3 deviance state."
        )
    return float(dev)


def _gaussian_penalty_quadratic(model, sol, sp) -> float:
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

    method_u = str(method).upper()
    if method_u not in {"ML", "REML", "LAML"}:
        raise NotImplementedError(
            "Exact dynamic Gaussian derivatives are currently implemented only for "
            "ML/REML/LAML."
        )

    n_s = int(model.n_samples_)
    w1 = prior_weights_diagonal_from_fit(sol, n_s)
    dev = _gaussian_dynamic_deviance(sol, y, w1)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    Pq = float(kernel.bSb)
    F = float(dev) + float(Pq)
    nobs = float(model.n_samples_)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    n_weighted = float(nu + Mp)
    reml_ind = 1.0 if method_u in {"REML", "LAML"} else 0.0
    # Mirror `mgcv/R/gam.fit3.r` `scoreType %in c("REML","ML")` branch:
    # profiled Gaussian scale solves F / (n_w - gamma * remlInd * Mp).
    prof_df = n_weighted - gamma * reml_ind * Mp
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

    n_sp = int(_n_smoothing_params(model) or 0)
    F1 = np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64)
    F2 = np.asarray(kernel.D2 + kernel.bSb2, dtype=np.float64)
    K1 = np.asarray(kernel.K1, dtype=np.float64)
    K2 = np.asarray(kernel.K2, dtype=np.float64)
    grad_full = 0.5 * coeff * F1 / F + K1
    hess_full = 0.5 * coeff * (F2 / F - np.outer(F1, F1) / (F * F)) + K2

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
        "K1_free": np.asarray(K1[free_mask], dtype=np.float64).copy(),
        "K2_free": np.asarray(
            K2[np.ix_(free_mask, free_mask)], dtype=np.float64
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
    if method_u not in {"ML", "REML", "LAML"}:
        raise ValueError(
            "method must be 'ML', 'REML', or 'LAML' for the joint Gaussian path."
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
    nu, sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    dev = _gaussian_dynamic_deviance(sol, y, w1)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    Pq = float(kernel.bSb)
    rss_bSb = float(dev) + float(Pq)
    if not np.isfinite(rss_bSb) or rss_bSb <= 0.0:
        return np.inf
    sigma2 = float(np.exp(float(log_sigma2)))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return np.inf
    det_term = float(kernel.K)
    if not np.isfinite(det_term):
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
    reml_ind = 1.0 if method_u in {"REML", "LAML"} else 0.0
    return (
        (rss_bSb / (2.0 * sigma2) - ls0) / gamma
        + det_term
        - reml_ind * (Mp / 2.0) * (np.log(2.0 * np.pi * sigma2) - np.log(gamma))
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
    if method_u not in {"ML", "REML", "LAML"}:
        raise ValueError(
            "method must be 'ML', 'REML', or 'LAML' for the profiled Gaussian path."
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
    reml_ind = 1.0 if method_u in {"REML", "LAML"} else 0.0
    prof_df = float(nu + Mp - gamma * reml_ind * Mp)
    if (
        not np.isfinite(gamma)
        or gamma <= 0.0
        or not np.isfinite(prof_df)
        or prof_df <= 0.0
    ):
        return np.inf

    dev = _gaussian_dynamic_deviance(sol, y, w1)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method_u)
    Pq = float(kernel.bSb)
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
    nobs = float(model.n_samples_)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    dev = _gaussian_dynamic_deviance(sol, y, w1)
    F = float(dev) + float(kernel.bSb)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    gamma = float(model.score_gamma)
    n_weighted = float(nu + Mp)
    reml_ind = 1.0 if method_u in {"REML", "LAML"} else 0.0
    coeff = n_weighted / gamma - reml_ind * Mp if gamma > 0.0 else np.nan
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

    nobs = float(model.n_samples_)
    w1 = prior_weights_diagonal_from_fit(sol, int(nobs))
    dev = _gaussian_dynamic_deviance(sol, y, w1)
    F = float(dev) + float(kernel.bSb)
    Mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    n_eff = getattr(model, "n_true_", None)
    nu, _sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
        w1, nobs, Mp, n_effective_total=n_eff
    )
    n_weighted = float(nu + Mp)
    reml_ind = 1.0 if method_u in {"REML", "LAML"} else 0.0
    coeff = n_weighted / gamma - reml_ind * Mp if gamma > 0.0 else np.nan
    if not np.isfinite(F) or F <= 0.0:
        return None
    if not np.isfinite(coeff):
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
