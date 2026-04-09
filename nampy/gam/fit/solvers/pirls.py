"""
Entry points for the penalized IRLS solver for non-Gaussian GAMs.

:func:`solve_pirls_fit` is the model-level entry point called by the fitting
orchestrator.  It extracts design and penalty information from the model object
and delegates to :func:`fit_pirls_core`.
"""

import numpy as np

from ..state import FitCoreSolution
from .pirls_core import fit_pirls_core


def solve_pirls_gam(
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
    return fit_pirls_core(
        Z=Z,
        y=y,
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        family=family,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        max_step_halving=max_step_halving,
        offset=offset,
        weights=weights,
        coef_start=coef_start,
        etastart=etastart,
        mustart=mustart,
    )


def solve_pirls_fit(model, y, smoothing_params, weights=None):
    """
    Model-level entry point for the penalized IRLS solver.

    Extracts the design matrix, penalty blocks, and offset from the model,
    runs :func:`fit_pirls_core`, and stores the converged coefficient vector
    on the model for use as a warm start in subsequent iterations.

    When the family has ``estimate_theta=True``, EFS (Embedded Fisher Scoring)
    updates theta after each IRLS step inside :func:`fit_pirls_core`, matching
    mgcv's ``gam.fit4.r`` lines 507-515.

    Returns a :class:`~nampy.gam.fit.state.FitCoreSolution` wrapping the
    converged working system.
    """
    coef_start = getattr(model, "_pirls_eval_start_", None)
    if coef_start is None:
        coef_start = getattr(model, "_pirls_coef_start_", None)
    if coef_start is not None:
        coef_start = np.asarray(coef_start, dtype=np.float64).ravel()
        if coef_start.shape != (
            int(model.Z.shape[1] + (1 if model.fit_intercept else 0)),
        ):
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

    disable_theta_efs = bool(getattr(model, "_pirls_disable_theta_efs_", False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    if disable_theta_efs:
        model.family._disable_theta_efs = True

    try:
        sol = solve_pirls_gam(
            Z=model.Z,
            y=y,
            penalty_blocks=model.penalty_blocks_,
            smoothing_params=smoothing_params,
            family=model.family,
            fit_intercept=model.fit_intercept,
            max_iter=int(getattr(model, "max_irls_iter", 100)),
            tol=float(getattr(model, "irls_tol", 1e-8)),
            max_step_halving=int(getattr(model, "max_step_halving", 25)),
            offset=model.offset_train_,
            weights=weights,
            coef_start=coef_start,
            etastart=etastart,
            mustart=mustart,
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
