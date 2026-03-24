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
First derivatives (``deriv >= 1``): gradient of each score w.r.t. all
log-smoothing parameters.

Second derivatives (``deriv >= 2``): Hessian for GCV/GACV and UBRE.  The REML
Hessian requires an additional unknown-scale derivative block not yet
implemented; ``reml_hess_log_sp`` is left as ``None``.

Not yet implemented
-------------------
- P-REML / Pearson-Laplace and NCV score paths.
- Unknown-scale ``sigma^2`` derivative in the REML Hessian.
- Non-Gaussian families (those use the full P-IRLS derivative chain).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import cho_factor

from nampy.gam.smoothness.criteria.gaussian_reml_algebra import (
    gaussian_reml_laplace_score,
    gaussian_weighted_residual_sum_squares,
    prior_weights_diagonal_from_fit,
    profiled_gaussian_reml_variance,
    quadratic_form_penalty,
)
from nampy.gam.smoothness.criteria.laplace import _penalty_derivative_matrices
from nampy.gam.smoothness.criteria.penalty import (
    _static_penalty_null_dim,
    _stable_penalty_logdet,
    _stable_penalty_logdet_derivatives,
)
from nampy.gam.smoothness.criteria.pirls_reml_derivative_blocks import (
    _deviance_chained_to_smoothing,
    _deviance_coefficient_derivatives,
    _hat_matrix_trace_and_sp_derivatives,
    _logdet_penalized_system_derivatives,
    _penalty_quadratic_and_sp_derivatives,
)


def _score_type_bucket(score_type: str) -> str:
    st = str(score_type).upper().strip()
    if st in {"REML", "P-REML", "NCV"}:
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
        ``penalty_blocks_``, ``n_coef_``, ``fit_intercept``, ``family`` (Gaussian),
        ``n_samples_``, optional ``n_true_``.
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
        ``gcv_score``, ``ubre_score`` (``None`` if not applicable), optional
        ``*_grad_log_sp``, ``*_hess_log_sp``, and intermediate derivative tensors
        ``D1``, ``D2``, ``logdet_a1``, ``logdet_a2``, ``logdet_s1``, ``logdet_s2``,
        ``tr_a1``, ``tr_a2`` (GCV branch) or REML combinations (layout aligned with mgcv).
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
    sol = model._solve_gaussian_given_smoothing(y, sp)

    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64).ravel()
    eta = np.asarray(sol["eta"], dtype=np.float64).ravel()
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

    n_true = float(getattr(model, "n_true_", n_s))
    nobs = float(n_s)
    denom_scale = n_true - tr_a
    if not np.isfinite(denom_scale) or denom_scale <= 0.0:
        scale_est = float("nan")
    else:
        scale_est = float((dev + pen) / denom_scale)

    ldet_a = sol.get("log_det_XtWX_plus_penalty", None)
    if ldet_a is not None and np.isfinite(float(ldet_a)):
        log_det_xtwx_plus_penalty = float(ldet_a)
    else:
        try:
            cA, loA = cho_factor(A, check_finite=False)
            log_det_xtwx_plus_penalty = 2.0 * float(
                np.sum(np.log(np.abs(np.diag(cA))))
            )
        except np.linalg.LinAlgError:
            log_det_xtwx_plus_penalty = float("nan")
    if not np.isfinite(log_det_xtwx_plus_penalty):
        raise ValueError(
            "Could not obtain finite log|X'WX+S| for Gaussian post-processing."
        )

    log_det_penalty_stable = float(_stable_penalty_logdet(model, sp))
    ldiff = log_det_xtwx_plus_penalty - log_det_penalty_stable

    Mp = float(
        _static_penalty_null_dim(model)
        + int(bool(getattr(model, "fit_intercept", False)))
    )

    gamma_eff = float(model.score_gamma if gamma is None else gamma)
    if not np.isfinite(gamma_eff) or gamma_eff <= 0.0:
        raise ValueError("gamma must be finite and positive.")

    bucket = _score_type_bucket(score_type)
    reml_score = gcv_score = ubre_score = None
    reml_score_profiled_scale = None
    reml_like = bucket in {"REML", "ML"}
    if reml_like:
        if not np.isfinite(scale_est) or scale_est <= 0.0:
            reml_score = float("nan")
        else:
            reml_score = gaussian_reml_laplace_score(
                dev,
                pen,
                scale_est,
                ldiff,
                Mp,
                w,
                gamma=gamma_eff,
                reml=(bucket == "REML"),
            )
        scale_prof = profiled_gaussian_reml_variance(dev, pen, n_true, Mp)
        if np.isfinite(scale_prof) and scale_prof > 0.0:
            reml_score_profiled_scale = gaussian_reml_laplace_score(
                dev,
                pen,
                float(scale_prof),
                ldiff,
                Mp,
                w,
                gamma=gamma_eff,
                reml=(bucket == "REML"),
            )

    if bucket in {"GCV", "GACV"}:
        delta = nobs - gamma_eff * tr_a
        if np.isfinite(delta) and abs(delta) > 1e-300:
            gcv_score = float(nobs * dev / (delta**2))
        else:
            gcv_score = float("inf")

    if bucket == "UBRE":
        delta = nobs - gamma_eff * tr_a
        sc = float(known_scale) if known_scale is not None else scale_est
        if np.isfinite(delta) and np.isfinite(sc):
            ubre_score = float(dev / nobs - 2.0 * delta * sc / nobs + sc)
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

    n_sp = int(model.n_smoothing_params_ or 0)
    P_derivs = _penalty_derivative_matrices(model, sp)
    dev_grad, dev_hess = _deviance_coefficient_derivatives(
        model, yv, eta, mu, w, X
    )

    dbeta: list[np.ndarray] = []
    dA: list[np.ndarray] = []
    dXtWX: list[np.ndarray] = []
    d2beta_mat: list[list[np.ndarray | None]] = [
        [None] * n_sp for _ in range(n_sp)
    ]
    d2A_mat: list[list[np.ndarray | None]] = [[None] * n_sp for _ in range(n_sp)]
    d2XtWX_mat: list[list[np.ndarray | None]] = [
        [None] * n_sp for _ in range(n_sp)
    ]

    for j in range(n_sp):
        Pj = P_derivs[j]
        if not np.any(Pj):
            z = np.zeros_like(beta, dtype=np.float64)
            dbeta.append(z)
            dA.append(np.zeros_like(A, dtype=np.float64))
            dXtWX.append(np.zeros_like(XtWX, dtype=np.float64))
            continue
        dbj = -(A_inv @ (Pj @ beta))
        dbeta.append(dbj)
        dA.append(np.asarray(Pj, dtype=np.float64))
        dXtWX.append(np.zeros_like(XtWX, dtype=np.float64))

    if deriv >= 2:
        for j in range(n_sp):
            Pj = P_derivs[j]
            for k in range(j, n_sp):
                Pk = P_derivs[k]
                delta_jk = 1.0 if j == k else 0.0
                d2beta_jk = -(
                    A_inv
                    @ (
                        Pk @ dbeta[j]
                        + Pj @ dbeta[k]
                        + delta_jk * (Pj @ beta)
                    )
                )
                d2beta_mat[j][k] = d2beta_jk
                d2beta_mat[k][j] = d2beta_jk
                d2XtWX_jk = np.zeros_like(XtWX, dtype=np.float64)
                d2XtWX_mat[j][k] = d2XtWX_jk
                d2XtWX_mat[k][j] = d2XtWX_jk
                if j == k and np.any(Pj):
                    d2A_jk = d2XtWX_jk + np.asarray(Pj, dtype=np.float64)
                else:
                    d2A_jk = d2XtWX_jk
                d2A_mat[j][k] = d2A_jk
                d2A_mat[k][j] = d2A_jk
    elif deriv == 1:
        zb = np.zeros_like(beta, dtype=np.float64)
        zA = np.zeros_like(A, dtype=np.float64)
        for j in range(n_sp):
            for k in range(j, n_sp):
                d2beta_mat[j][k] = zb
                d2beta_mat[k][j] = zb
                zxt = np.zeros_like(XtWX, dtype=np.float64)
                d2XtWX_mat[j][k] = zxt
                d2XtWX_mat[k][j] = zxt
                d2A_mat[j][k] = zA
                d2A_mat[k][j] = zA

    D1_dev, D2_dev = _deviance_chained_to_smoothing(
        dev_grad, dev_hess, dbeta, d2beta_mat
    )
    bSb, bSb1, bSb2 = _penalty_quadratic_and_sp_derivatives(
        beta=beta,
        P_total=P_tot,
        P_derivs=P_derivs,
        dbeta_cols=dbeta,
        d2beta_mat=d2beta_mat,
    )
    # Penalized deviance D_p = dev + b'Sb — used in the REML score combination.
    D1_pen = D1_dev + bSb1
    D2_pen = D2_dev + bSb2 if deriv >= 2 else None

    logdet_a1, logdet_a2 = _logdet_penalized_system_derivatives(A_inv, dA, d2A_mat)
    tr_hat, tr_a1, tr_a2 = _hat_matrix_trace_and_sp_derivatives(
        A_inv,
        XtWX,
        dA,
        d2A_mat,
        dXtWX,
        d2XtWX_mat,
    )

    _ldS0, logdet_s1, logdet_s2 = _stable_penalty_logdet_derivatives(
        model, sp, order=2 if deriv >= 2 else 1
    )

    out.update(
        {
            "D1_deviance": D1_dev.copy(),
            "D1_penalized_deviance": D1_pen.copy(),
            "D1": D1_pen.copy(),
            "penalized_deviance_grad": D1_pen.copy(),
            "bSb": bSb,
            "logdet_a1": logdet_a1.copy(),
            "logdet_s1": logdet_s1.copy(),
            "tr_a_hat": float(tr_hat),
            "tr_a1_gcv": tr_a1.copy(),
        }
    )
    if deriv >= 2 and D2_pen is not None:
        out["D2_deviance"] = D2_dev.copy()
        out["D2_penalized_deviance"] = D2_pen.copy()
        out["D2"] = D2_pen.copy()
        out["logdet_a2"] = logdet_a2.copy()
        out["logdet_s2"] = logdet_s2.copy()
        out["tr_a2_gcv"] = tr_a2.copy()

    scale = scale_est
    if not np.isfinite(scale) or scale <= 0.0:
        out["reml_grad_log_sp"] = None
        out["reml_hess_log_sp"] = None
        out["gcv_grad_log_sp"] = None
        out["gcv_hess_log_sp"] = None
        return out

    if deriv >= 1:
        if reml_like:
            reml1 = D1_pen / (2.0 * scale * gamma_eff) + 0.5 * logdet_a1 - 0.5 * logdet_s1
            out["reml_grad_log_sp"] = reml1.copy()
        if bucket in {"GCV", "GACV"}:
            delta = nobs - gamma_eff * tr_a
            delta2 = delta * delta
            delta3 = delta * delta2
            gcv1 = nobs * D1_dev / delta2 + 2.0 * nobs * dev * tr_a1 * gamma_eff / delta3
            out["gcv_grad_log_sp"] = gcv1.copy()
        if bucket == "UBRE":
            sc = float(known_scale) if known_scale is not None else scale
            delta = nobs - gamma_eff * tr_a
            ubre1 = D1_dev / nobs + gamma_eff * tr_a1 * 2.0 * sc / nobs
            out["ubre_grad_log_sp"] = ubre1.copy()

    if deriv >= 2:
        # The REML Hessian requires an additional unknown-scale branch for estimated
        # sigma^2 that is not yet implemented; reml_hess_log_sp is left as None.
        # GCV/GACV second derivatives use only the deviance second derivative D2.
        if reml_like:
            out["reml_hess_log_sp"] = None
        if bucket in {"GCV", "GACV"} and D2_pen is not None:
            delta = nobs - gamma_eff * tr_a
            delta2 = delta * delta
            delta3 = delta * delta2
            outer_td = np.outer(tr_a1, D1_dev)
            gcv2 = (
                (outer_td + outer_td.T) * gamma_eff * 2.0 * nobs / delta3
                + 6.0 * nobs * dev * np.outer(tr_a1, tr_a1) * gamma_eff**2 / (delta2 * delta2)
                + nobs * D2_dev / delta2
                + 2.0 * nobs * dev * gamma_eff * tr_a2 / delta3
            )
            out["gcv_hess_log_sp"] = gcv2.copy()
        if bucket == "UBRE" and D2_pen is not None:
            sc = float(known_scale) if known_scale is not None else scale
            out["ubre_hess_log_sp"] = D2_dev / nobs + 2.0 * gamma_eff * tr_a2 * sc / nobs

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
