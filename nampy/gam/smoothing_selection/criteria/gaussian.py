"""
GCV and Gaussian ML/REML/LAML smoothing_selection-selection criteria.

Provides two code paths:

- **Exact** (``criterion_ml_reml_exact``): Laplace REML/ML using the
  reparameterised mixed-model form.  Fast and precise for most smooth designs;
  may return ``inf`` for designs where the fixed-effects sub-matrix is singular
  (e.g. random-effect plus intercept).

- **Dynamic** (``criterion_ml_reml_exact_dynamic``): Fully profiled Gaussian
  REML/ML objective evaluated directly from the penalized least-squares solve
  without the mixed-model reparameterisation.  More robust for designs with
  collinear fixed effects.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_profiled,
)
from .laplace import _laplace_lambda_vector
from .penalty import _static_fixed_and_random_designs


def gcv_score_gaussian(model, y, log_smoothing_params):
    sp = model._expand_smoothing_params_from_log(log_smoothing_params)
    sol = model._solve_gaussian_given_smoothing(y, sp)
    n = model.n_samples_
    den = 1.0 - model.score_gamma * sol["trace_H"] / n
    if den <= 1e-12 or not np.isfinite(den):
        return np.inf
    return (sol["rss"] / n) / (den**2)


def criterion_gcv_gaussian(model, y, log_sp):
    return gcv_score_gaussian(model, y, log_sp)


def criterion_ml_reml_exact(model, y, log_sp, method):
    """
    Laplace / Schur-complement ML and REML for the reparameterized mixed form.

    For ``bs=\"re\"`` with a parametric intercept, ``col(1) ⊆ col(Z)`` makes
    ``Xf' K Xf`` singular here, so this Laplace path returns ``inf`` at many
    ``(y, sp)``. Those models are scored for ML/REML **outer selection** via
    ``criterion_ml_reml_exact_dynamic`` (Wood ``X'WX+S`` / ``\\log|A|-\\log|S|``)
    instead; see ``nampy.gam.smoothing_selection.criteria.ml_reml``.
    """
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

    logdet_XtKX = 0.0 if p == 0 else 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return (n - p) * np.log(rss_v / (n - p)) + logdet_Vtilde + logdet_XtKX


def criterion_ml_reml_gaussian_exact_joint(
    model, y, log_sp_free, log_sigma2, method="REML"
):
    """
    Joint (log sp, log sigma^2) Gaussian REML/LAML for the `gaussian_exact` backend.

    Delegates to `criterion_ml_reml_gaussian_dynamic_joint`, i.e. the Wood-style
    penalized likelihood using ``A``, ``log|A| - log|S|``, and ``nu * log(2 pi sigma^2)``
    with ``rss_bSb / sigma^2``. That matches the standard Gaussian REML outer objective
    (the same target the dynamic backend minimises jointly with scale).

    A previous Laplace ``Z'Z + lambda`` formulation coincided with the *profiled*
    ``criterion_ml_reml_exact`` score, but it is not the unconcentrated joint
    objective and it used ``max(rss, 1e-14)``, which breaks down when optimal
    ``sigma^2`` is far below that floor (e.g. near-interpolating MRF fits).
    """
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the joint Gaussian exact REML path."
        )

    if not model._can_use_exact_gaussian_ml_reml():
        raise NotImplementedError(
            "Joint exact Gaussian REML is only available when the exact Gaussian ML/REML "
            "structure applies."
        )

    method_u = str(method).upper()
    if method_u not in {"REML", "LAML"}:
        raise ValueError("method must be 'REML' or 'LAML'.")

    return criterion_ml_reml_gaussian_dynamic_joint(
        model, y, log_sp_free, log_sigma2, method=method_u
    )


def criterion_ml_reml_exact_dynamic(model, y, log_sp, method):
    if abs(model.score_gamma - 1.0) > 1e-12:
        raise NotImplementedError(
            "score_gamma != 1 is not yet implemented for the dynamic Gaussian ML/REML/LAML path."
        )
    y = model.family.validate_y(y)
    y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
    sp = model._expand_smoothing_params_from_log(log_sp)

    sol = model._solve_gaussian_given_smoothing(y, sp)
    if method in {"REML", "LAML"}:
        return criterion_ml_reml_gaussian_dynamic_profiled(
            model,
            y,
            log_sp,
            method=method,
        )

    X = np.asarray(sol["X"], dtype=np.float64)
    Xf, Zr, split = _static_fixed_and_random_designs(model, X, sp)
    n = X.shape[0]
    p = int(Xf.shape[1])
    q = int(Zr.shape[1])

    if q == 0:
        rss_v = max(float(sol["rss"]), 1e-14)
        if method == "ML":
            return n * np.log(rss_v / n)

        if n <= p:
            return np.inf

        if p == 0:
            return (n - p) * np.log(rss_v / (n - p))

        XtX_fix = Xf.T @ Xf
        try:
            cFix, _ = cho_factor(XtX_fix, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        return (n - p) * np.log(rss_v / (n - p)) + logdet_fix

    M = Zr.T @ Zr + np.eye(q, dtype=np.float64)
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
            cXKX, _ = cho_factor(XtKX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        XtKy = Xf.T @ Ky
        b_hat = cho_solve((cXKX, False), XtKy, check_finite=False)
        rss_v = max(float(y_eff @ Ky - XtKy @ b_hat), 1e-14)
    else:
        cXKX = None
        rss_v = max(float(y_eff @ Ky), 1e-14)

    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(cM)))))
    logdet_Vtilde = logdet_M - float(split["logdet_plus"])

    if method == "ML":
        return n * np.log(rss_v / n) + logdet_Vtilde

    if n <= p:
        return np.inf

    logdet_XtKX = 0.0 if p == 0 else 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return (n - p) * np.log(rss_v / (n - p)) + logdet_Vtilde + logdet_XtKX
