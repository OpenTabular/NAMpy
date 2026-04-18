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

from ..._model_state import _design_matrix, _term_blocks_seq
from ...fit.model_ops import (
    can_use_exact_gaussian_ml_reml,
    expand_smoothing_params_from_log,
    solve_gaussian_given_smoothing,
)
from ...fit.penalized_system import build_full_design
from ..reparam import _static_fixed_and_random_designs, dynamic_reparam_design
from .gaussian_dyn import criterion_ml_reml_gaussian_dynamic_profiled
from .gaussian_reml_algebra import gaussian_reml_saturation_terms_wrt_variance


def gcv_score_gaussian(model, y, log_smoothing_params):
    sp = expand_smoothing_params_from_log(model, log_smoothing_params)
    sol = solve_gaussian_given_smoothing(model, y, sp)
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
    if not can_use_exact_gaussian_ml_reml(model):
        raise NotImplementedError(
            "Exact Gaussian ML/REML/LAML is currently available only for "
            "terms whose penalties do not couple disconnected support "
            "components through null-space penalties."
        )

    # `mgcv`'s fs constructor (`smooth.construct.fs.smooth.spec`) remains in the
    # direct coefficient parameterization with replicated null-space penalties.
    # Our current mixed-model Laplace port is not exact for that surface, while
    # the Wood-style dynamic profile below matches mgcv on the audited parity
    # slices. Fail closed here so `criterion_ml_reml()` falls back to the
    # dynamic path instead of trusting a finite but wrong exact score.
    if any(
        str(getattr(tb, "term_type", "")).lower() == "factor_smooth_fs"
        for tb in _term_blocks_seq(model)
    ):
        return np.inf

    y = model.family.validate_y(y)
    y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
    sp = expand_smoothing_params_from_log(model, log_sp)
    X = build_full_design(_design_matrix(model), fit_intercept=model.fit_intercept)
    design = dynamic_reparam_design(model, X, sp)
    Xf = design.X_fix
    Zr = design.Z_rand
    n = Xf.shape[0]
    p = int(Xf.shape[1])
    q = int(Zr.shape[1])
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    reml_ind = 1.0 if method in {"REML", "LAML"} else 0.0
    w = np.ones(int(n), dtype=np.float64)

    if q == 0:
        if p == 0:
            rss_v = max(float(y_eff @ y_eff), 1e-14)
            scale = rss_v / float(n)
            ls = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
            return (rss_v / (2.0 * scale) - ls) / gamma

        XtX = Xf.T @ Xf
        try:
            cXtX, lo = cho_factor(XtX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        b_hat = cho_solve((cXtX, lo), Xf.T @ y_eff, check_finite=False)
        resid = y_eff - Xf @ b_hat
        rss_v = max(float(resid @ resid), 1e-14)
        prof_df = float(n - p * gamma)
        if method == "ML":
            prof_df = float(n)
        if not np.isfinite(prof_df) or prof_df <= 0.0:
            return np.inf
        scale = rss_v / prof_df
        if not np.isfinite(scale) or scale <= 0.0:
            return np.inf
        ls = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
        score = (rss_v / (2.0 * scale) - ls) / gamma

        if method == "ML":
            return score

        logdet_XtX = 2.0 * float(np.sum(np.log(np.diag(cXtX))))
        return score + logdet_XtX / 2.0 - reml_ind * p * (
            np.log(2.0 * np.pi * scale) / 2.0 - np.log(gamma) / 2.0
        )

    M = design.ZtZ_rand + np.eye(q, dtype=np.float64)
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
    logdet_Vtilde = logdet_M
    prof_df = float(n - p * gamma)
    if method == "ML":
        prof_df = float(n)
    if not np.isfinite(prof_df) or prof_df <= 0.0:
        return np.inf
    scale = rss_v / prof_df
    if not np.isfinite(scale) or scale <= 0.0:
        return np.inf
    ls = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
    score = (rss_v / (2.0 * scale) - ls) / gamma + logdet_Vtilde / 2.0

    if method == "ML":
        return score

    logdet_XtKX = 0.0 if p == 0 else 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return score + logdet_XtKX / 2.0 - reml_ind * p * (
        np.log(2.0 * np.pi * scale) / 2.0 - np.log(gamma) / 2.0
    )


def criterion_ml_reml_exact_dynamic(model, y, log_sp, method):
    y = model.family.validate_y(y)
    y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
    sp = expand_smoothing_params_from_log(model, log_sp)
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    reml_ind = 1.0 if method in {"REML", "LAML"} else 0.0
    w = np.ones(int(y_eff.shape[0]), dtype=np.float64)

    sol = solve_gaussian_given_smoothing(model, y, sp)
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
        prof_df = float(n - p * gamma)
        if method == "ML":
            prof_df = float(n)
        if prof_df <= 0.0:
            return np.inf
        scale = rss_v / prof_df
        if not np.isfinite(scale) or scale <= 0.0:
            return np.inf
        ls = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
        score = (rss_v / (2.0 * scale) - ls) / gamma

        if p == 0:
            return score

        XtX_fix = Xf.T @ Xf
        try:
            cFix, _ = cho_factor(XtX_fix, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        logdet_fix = 2.0 * float(np.sum(np.log(np.abs(np.diag(cFix)))))
        return score + logdet_fix / 2.0 - reml_ind * p * (
            np.log(2.0 * np.pi * scale) / 2.0 - np.log(gamma) / 2.0
        )

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
    prof_df = float(n - p * gamma)
    if method == "ML":
        prof_df = float(n)
    if prof_df <= 0.0:
        return np.inf
    scale = rss_v / prof_df
    if not np.isfinite(scale) or scale <= 0.0:
        return np.inf
    ls = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
    score = (rss_v / (2.0 * scale) - ls) / gamma + logdet_Vtilde / 2.0

    if method == "ML":
        return score

    logdet_XtKX = 0.0 if p == 0 else 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
    return score + logdet_XtKX / 2.0 - reml_ind * p * (
        np.log(2.0 * np.pi * scale) / 2.0 - np.log(gamma) / 2.0
    )
