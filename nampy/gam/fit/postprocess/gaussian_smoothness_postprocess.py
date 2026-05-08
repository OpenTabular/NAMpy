"""
Post-fit smoothness scores and derivatives for Gaussian GAMs.

After the penalized IRLS solver has converged with fixed smoothing parameters,
this module computes the REML / ML / GCV / UBRE / AIC smoothness-selection
scores and their first- and second-order derivatives with respect to
log-smoothing parameters.

Gaussian Fisher identity
------------------------
For Gaussian models with an identity link the IRLS working weights are the
constant prior weights, so the implicit-function derivatives of the fitted
coefficients with respect to smoothing parameters reduce to simple linear
algebra involving ``A = X'WX + S`` and its inverse.  This is the case handled
here.  Non-Gaussian families require the full penalized IRLS derivative chain.

Derivative computation
----------------------
Gradients / Hessians are delegated to ``nampy.gam.smoothing_selection.criteria``.

Not yet implemented
-------------------
- P-REML / Pearson-Laplace score paths.
- Non-Gaussian families (those use the full P-IRLS derivative chain).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import cho_factor

from nampy.gam._mgcv_constants import LOG_GUARD_MIN
from nampy.gam._model_state import (
    _coef_column_offset,
    _fit_result,
    _fit_state,
    _n_smoothing_params,
)
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
)
from nampy.gam.smoothing_selection.criteria.gaussian import criterion_gcv_gaussian
from nampy.gam.smoothing_selection.criteria.gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_profiled,
)
from nampy.gam.smoothing_selection.criteria.gaussian_reml_algebra import (
    gaussian_weighted_residual_sum_squares,
    prior_weights_diagonal_from_fit,
    quadratic_form_penalty,
)
from nampy.gam.smoothing_selection.criteria.ml_reml import (
    criterion_ml_reml,
    resolve_ml_reml_scoring_backend,
)
from nampy.gam.smoothing_selection.criteria.pirls import criterion_ubre_pirls
from nampy.gam.smoothing_selection.reparam import (
    _stable_penalty_logdet,
    _static_penalty_null_dim,
)


def refresh_gaussian_ml_reml_score_from_fit_state(model: Any, y: np.ndarray) -> None:
    """
    Recompute reported Gaussian ML/REML criterion from fitted state when needed.

    This mirrors mgcv's reported criterion scale for the exact Gaussian backend.
    """
    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml"}:
        return
    if str(getattr(getattr(model, "family", None), "name", "")).lower() != "gaussian":
        return
    if getattr(model, "_gaussian_reml_sigma2_opt_", None) is not None:
        return

    fit_state = _fit_state(model)
    fit_result = _fit_result(model)
    if fit_state is None:
        return

    try:
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if backend != "gaussian_exact":
            return

        fixed_mask = (
            np.zeros(_n_smoothing_params(model), dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_vals = np.asarray(model.smoothing_params[~fixed_mask], dtype=np.float64)
        log_free = (
            np.log(np.maximum(free_vals, LOG_GUARD_MIN))
            if free_vals.size > 0
            else np.empty((0,), dtype=np.float64)
        )

        n_s = int(model.n_samples_)
        yv = np.asarray(y, dtype=np.float64).ravel()
        mu_v = np.asarray(fit_result.mu, dtype=np.float64).ravel()
        w = prior_weights_diagonal_from_fit(fit_state, n_s)
        dev = gaussian_weighted_residual_sum_squares(yv, mu_v, w)
        p_pen = (
            float(fit_state.penalty_quadratic)
            if fit_state.penalty_quadratic is not None
            else quadratic_form_penalty(
                np.asarray(fit_result.coef_full, dtype=np.float64),
                np.asarray(fit_state.penalty_matrix, dtype=np.float64),
            )
        )
        mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
        nu = float(n_s) - mp
        if not (np.isfinite(nu) and nu > 0.0):
            return

        sigma2_prof = (dev + p_pen) / nu
        if not (np.isfinite(sigma2_prof) and sigma2_prof > 0.0):
            return

        log_s2 = float(np.log(sigma2_prof))
        branch_m = "REML" if method == "reml" else "ML"
        wood = float(
            criterion_ml_reml_gaussian_dynamic_joint(
                model, yv, log_free, log_s2, method=branch_m
            )
        )
        if np.isfinite(wood):
            model.smoothing_score_ = wood
    except Exception:
        return

    # Keep the private joint REML scale for criterion evaluation, but do not
    # overwrite the public fit scale/covariances. mgcv reports the final
    # deviance-based `sig2` on the fit object even when the outer optimiser
    # carried a separate Gaussian scale parameter.


def _score_type_bucket(score_type: str) -> str:
    st = str(score_type).upper().strip()
    if st in {"REML", "P-REML"}:
        return "REML"
    if st in {"ML", "P-ML"}:
        return "ML"
    if st in {"GCV", "GACV"}:
        return "GCV"
    if st in {"UBRE", "AIC", "UBREAIC"}:
        return "UBRE"
    raise ValueError(
        f"Unsupported score_type {score_type!r} for Gaussian post-processing "
        "(use REML, ML, GCV, GACV, UBRE, AIC)."
    )


def gaussian_smoothness_postprocess(
    model: Any,
    y: np.ndarray,
    smoothing_params: np.ndarray,
    *,
    score_type: str = "REML",
    deriv: int = 0,
    gamma: float | None = None,
    known_scale: float | None = None,
) -> dict[str, Any]:
    """
    Post-fit Gaussian smoothness scores (and optional derivatives).

    Parameters
    ----------
    model
        Fitted ``GAM``-like object with ``_solve_gaussian_given_smoothing``,
        compiled penalties / coefficient count, ``fit_intercept``, ``family``
        (Gaussian), ``n_samples_``, optional ``n_true_``.
    y
        Response vector used for the fit.
    smoothing_params
        Linear-scale smoothing parameters :math:`\\lambda` (same as ``model.sp`` in mgcv).
    score_type
        ``REML``, ``ML``, ``GCV``, ``GACV``, ``UBRE`` / ``AIC`` (case-insensitive).
    deriv
        ``0``, ``1``, or ``2``.  When positive, fills gradient/Hessian w.r.t. **all**
        ``log(sp)`` components; apply your fixed-sp mask outside if needed.
    gamma
        GCV inflation factor (defaults to ``model.score_gamma``).
    known_scale
        Known scale :math:`\\sigma^2` for UBRE/AIC (``gam``'s ``scale`` argument).
        If ``None``, UBRE uses the deviance scale estimate
        ``(dev + penalty) / (n_{\\mathrm{true}} - \\mathrm{tr}(A))``.

    Returns
    -------
    dict
        ``tr_a``, ``pearson_chi2``, ``scale_est``, ``deviance``, ``penalty_quadratic``,
        ``log_det_xtwx_plus_penalty``, ``log_det_penalty_stable``, ``reml_score``,
        ``gcv_score``, ``ubre_score`` (``None`` if not applicable), plus optional
        criterion gradients / Hessians with respect to ``log(sp)``.
    """
    if deriv not in (0, 1, 2):
        raise ValueError("deriv must be 0, 1, or 2.")

    fam = getattr(model, "family", None)
    if fam is None or str(getattr(fam, "name", "")).lower() != "gaussian":
        raise NotImplementedError(
            "gaussian_smoothness_postprocess is implemented for Gaussian families only."
        )

    y = fam.validate_y(y)
    sp = np.asarray(smoothing_params, dtype=np.float64).ravel()
    from ..backends import solve_gaussian_given_smoothing

    sol = solve_gaussian_given_smoothing(model, y, sp)

    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64).ravel()
    mu = np.asarray(sol["mu"], dtype=np.float64).ravel()
    A = np.asarray(sol["A"], dtype=np.float64)
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    XtWX = np.asarray(sol["XtWX"], dtype=np.float64)
    P_tot = np.asarray(sol["penalty_matrix"], dtype=np.float64)

    n_s = int(model.n_samples_)
    w = prior_weights_diagonal_from_fit(sol, n_s)
    yv = np.asarray(y, dtype=np.float64).ravel()

    dev = gaussian_weighted_residual_sum_squares(yv, mu, w)
    pen = quadratic_form_penalty(beta, P_tot)

    H_coef = np.asarray(sol.get("coef_hat_matrix"), dtype=np.float64)
    if H_coef.size and H_coef.shape == (X.shape[1], X.shape[1]):
        tr_a = float(np.trace(H_coef))
    else:
        tr_a = float(np.trace(A_inv @ XtWX))

    pearson_chi2 = float(np.sum(w * (yv - mu) ** 2))

    nobs = float(n_s)
    scale_est = fam.estimate_dispersion(yv, mu, edf=tr_a, weights=w)

    ldet_a = sol.get("log_det_XtWX_plus_penalty", None)
    if ldet_a is not None and np.isfinite(float(ldet_a)):
        log_det_xtwx_plus_penalty = float(ldet_a)
    else:
        try:
            cA, loA = cho_factor(A, check_finite=False)
            log_det_xtwx_plus_penalty = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
        except np.linalg.LinAlgError:
            log_det_xtwx_plus_penalty = float("nan")
    if not np.isfinite(log_det_xtwx_plus_penalty):
        raise ValueError(
            "Could not obtain finite log|X'WX+S| for Gaussian post-processing."
        )

    log_det_penalty_stable = float(_stable_penalty_logdet(model, sp))
    gamma_eff = float(model.score_gamma if gamma is None else gamma)
    if not np.isfinite(gamma_eff) or gamma_eff <= 0.0:
        raise ValueError("gamma must be finite and positive.")

    bucket = _score_type_bucket(score_type)
    log_sp = np.log(np.maximum(sp, np.finfo(np.float64).tiny))
    reml_score = gcv_score = ubre_score = None
    reml_score_profiled_scale = None
    reml_like = bucket in {"REML", "ML"}
    if reml_like:
        reml_score = float(criterion_ml_reml(model, y, log_sp, bucket))
        reml_score_profiled_scale = float(
            criterion_ml_reml_gaussian_dynamic_profiled(
                model,
                y,
                log_sp,
                method=bucket,
            )
        )

    if bucket in {"GCV", "GACV"}:
        gcv_score = float(criterion_gcv_gaussian(model, y, log_sp))

    if bucket == "UBRE":
        if known_scale is not None:
            old_known_scale = getattr(model.family, "known_scale", None)
            model.family.known_scale = float(known_scale)
            try:
                ubre_score = float(criterion_ubre_pirls(model, y, log_sp))
            finally:
                model.family.known_scale = old_known_scale
        else:
            delta = nobs - gamma_eff * tr_a
            if np.isfinite(delta) and np.isfinite(scale_est):
                ubre_score = float(
                    dev / nobs - 2.0 * delta * scale_est / nobs + scale_est
                )
            else:
                ubre_score = float("nan")

    out: dict[str, Any] = {
        "tr_a": tr_a,
        "pearson_chi2": pearson_chi2,
        "scale_est": scale_est,
        "deviance": float(dev),
        "penalty_quadratic": float(pen),
        "log_det_xtwx_plus_penalty": log_det_xtwx_plus_penalty,
        "log_det_penalty_stable": log_det_penalty_stable,
        "reml_score": reml_score,
        "reml_score_profiled_scale": reml_score_profiled_scale,
        "gcv_score": gcv_score,
        "ubre_score": ubre_score,
        "score_type": score_type,
        "deriv": int(deriv),
    }

    if deriv == 0:
        return out

    if deriv >= 1:
        if reml_like:
            out["reml_grad_log_sp"] = np.asarray(
                criterion_gradient(model, y, log_sp, method=bucket.lower()),
                dtype=np.float64,
            )
        if bucket in {"GCV", "GACV"}:
            out["gcv_grad_log_sp"] = np.asarray(
                criterion_gradient(model, y, log_sp, method="gcv"),
                dtype=np.float64,
            )
        if bucket == "UBRE":
            if known_scale is not None:
                old_known_scale = getattr(model.family, "known_scale", None)
                model.family.known_scale = float(known_scale)
                try:
                    out["ubre_grad_log_sp"] = np.asarray(
                        criterion_gradient(model, y, log_sp, method="ubre"),
                        dtype=np.float64,
                    )
                finally:
                    model.family.known_scale = old_known_scale

    if deriv >= 2:
        if reml_like:
            out["reml_hess_log_sp"] = np.asarray(
                criterion_hessian(model, y, log_sp, method=bucket.lower()),
                dtype=np.float64,
            )
        if bucket in {"GCV", "GACV"}:
            out["gcv_hess_log_sp"] = np.asarray(
                criterion_hessian(model, y, log_sp, method="gcv"),
                dtype=np.float64,
            )
        if bucket == "UBRE" and known_scale is not None:
            old_known_scale = getattr(model.family, "known_scale", None)
            model.family.known_scale = float(known_scale)
            try:
                out["ubre_hess_log_sp"] = np.asarray(
                    criterion_hessian(model, y, log_sp, method="ubre"),
                    dtype=np.float64,
                )
            finally:
                model.family.known_scale = old_known_scale

    return out


def merge_gaussian_smoothness_into_fit_result(
    result: dict[str, Any],
    model: Any,
    y: np.ndarray,
    smoothing_params: np.ndarray,
    *,
    score_type: str = "REML",
    deriv: int = 2,
) -> dict[str, Any]:
    """Merge :func:`gaussian_smoothness_postprocess` outputs into a penalized-IRLS result dict."""
    diag = gaussian_smoothness_postprocess(
        model,
        y,
        smoothing_params,
        score_type=score_type,
        deriv=deriv,
    )
    merged = dict(result)
    for key in (
        "tr_a",
        "pearson_chi2",
        "scale_est",
        "deviance",
        "penalty_quadratic",
        "log_det_xtwx_plus_penalty",
        "log_det_penalty_stable",
        "reml_score",
        "reml_score_profiled_scale",
        "gcv_score",
        "ubre_score",
        "D1",
        "D1_deviance",
        "D1_penalized_deviance",
        "D2",
        "D2_deviance",
        "D2_penalized_deviance",
        "reml_grad_log_sp",
        "reml_hess_log_sp",
        "gcv_grad_log_sp",
        "gcv_hess_log_sp",
        "ubre_grad_log_sp",
        "ubre_hess_log_sp",
        "postproc",
    ):
        if key in diag:
            merged[key] = diag[key]
    merged["postproc"] = "gaussian_smoothness_postprocess"
    return merged
