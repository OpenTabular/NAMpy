"""
Model-level entry point for PIRLS fits.
"""

import numpy as np

from ..._model_state import (
    _coef_column_offset,
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _penalty_blocks_seq,
)
from ..linalg.stacked_qr import balanced_penalty_template_sqrt_for_rank
from ..penalized_system import build_full_design, build_full_penalty_from_blocks
from ..state import FitCoreSolution
from .irls_core import (
    _mgcv_effective_irls_tol,
    _mgcv_null_coef,
    _mgcv_poisson_identity_fisher_endpoint,
    irls_core,
)


def solve_pirls_fit(model, y, smoothing_params, weights=None):
    coef_start = getattr(model, "_pirls_eval_start_", None)
    if coef_start is None:
        coef_start = getattr(model, "_pirls_coef_start_", None)
    if coef_start is not None:
        coef_start = np.asarray(coef_start, dtype=np.float64).ravel()
        Z = np.asarray(_design_matrix(model), dtype=np.float64)
        if coef_start.shape != (int(Z.shape[1] + _coef_column_offset(model)),):
            coef_start = None

    etastart = getattr(model, "_pirls_eval_eta_start_", None)
    if etastart is None:
        etastart = getattr(model, "_pirls_eta_start_", None)
    if etastart is not None:
        etastart = np.asarray(etastart, dtype=np.float64).ravel()
        if etastart.shape != (int(model.n_samples_),):
            etastart = None

    mustart = getattr(model, "_pirls_eval_mu_start_", None)
    if mustart is None:
        mustart = getattr(model, "_pirls_mu_start_", None)
    if mustart is not None:
        mustart = np.asarray(mustart, dtype=np.float64).ravel()
        if mustart.shape != (int(model.n_samples_),):
            mustart = None

    fi = _fit_intercept(model)
    Z = np.asarray(_design_matrix(model), dtype=np.float64)
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_coef = _n_coef(model)
    X = build_full_design(Z, fit_intercept=fi)
    S = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        fit_intercept=fi,
        n_coef=n_coef,
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        penalty_blocks,
        fit_intercept=fi,
        n_coef=int(n_coef),
    )

    disable_theta_efs = bool(getattr(model, "_pirls_disable_theta_efs_", False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    if disable_theta_efs:
        model.family._disable_theta_efs = True

    force_stacked_qr = (
        str(getattr(model.family, "family_class", "")).lower() == "extended"
        and str(getattr(model.family, "name", "")).lower() == "negbin"
        and str(getattr(model.family, "link_name", "")).lower() == "log"
    )

    try:
        sol = irls_core(
            X,
            y,
            model.family,
            S,
            offset=model.offset_train_,
            weights=weights,
            fit_intercept=fi,
            max_iter=int(getattr(model, "max_irls_iter", 200)),
            tol=_mgcv_effective_irls_tol(
                model.family, float(getattr(model, "irls_tol", 1e-7))
            ),
            max_step_halving=int(getattr(model, "max_step_halving", 25)),
            coef_start=coef_start,
            null_coef=_mgcv_null_coef(X, y, model.family),
            etastart=etastart,
            mustart=mustart,
            fisher_scoring_only=_mgcv_poisson_identity_fisher_endpoint(model.family),
            penalty_rank_rows=rank_rows,
            force_stacked_qr=force_stacked_qr,
            near_singular_null_pin=("auto" if force_stacked_qr else False),
        )
    finally:
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)

    coef_out = np.asarray(sol["coef_full"], dtype=np.float64).copy()
    model._pirls_last_coef_ = coef_out
    model._pirls_last_eta_ = np.asarray(sol["eta"], dtype=np.float64).copy()
    model._pirls_last_mu_ = np.asarray(sol["mu"], dtype=np.float64).copy()
    model._pirls_last_inner_trace_ = list(sol.get("inner_trace", []) or [])
    if not bool(getattr(model, "_pirls_lock_start_", False)):
        model._pirls_coef_start_ = coef_out.copy()
        model._pirls_eta_start_ = np.asarray(sol["eta"], dtype=np.float64).copy()
        model._pirls_mu_start_ = np.asarray(sol["mu"], dtype=np.float64).copy()
    return FitCoreSolution.from_dict(sol)
