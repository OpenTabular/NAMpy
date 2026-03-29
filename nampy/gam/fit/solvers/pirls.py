"""
Entry points for the penalized IRLS solver for non-Gaussian GAMs.

:func:`solve_pirls_fit` is the model-level entry point called by the fitting
orchestrator.  It extracts design and penalty information from the model object
and delegates to :func:`fit_pirls_core`.
"""

import numpy as np

from .pirls_core import fit_pirls_core
from ..state import FitCoreSolution


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
        coef_start=coef_start,
        etastart=etastart,
        mustart=mustart,
    )


def solve_pirls_fit(model, y, smoothing_params):
    """
    Model-level entry point for the penalized IRLS solver.

    Extracts the design matrix, penalty blocks, and offset from the model,
    runs :func:`fit_pirls_core`, and stores the converged coefficient vector
    on the model for use as a warm start in subsequent iterations.

    For NegBin families with ``estimate_theta=True`` an outer loop alternates
    between solving for ``beta`` (inner PIRLS) and updating ``theta`` via
    Newton-Raphson MLE steps until both converge.

    Returns a :class:`~nampy.gam.fit.state.FitCoreSolution` wrapping the
    converged working system.
    """
    coef_start = getattr(model, "_pirls_eval_start_", None)
    if coef_start is None:
        coef_start = getattr(model, "_pirls_coef_start_", None)
    if coef_start is not None:
        coef_start = np.asarray(coef_start, dtype=np.float64).ravel()
        if coef_start.shape != (int(model.Z.shape[1] + (1 if model.fit_intercept else 0)),):
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

    family = model.family
    estimate_theta = bool(getattr(family, "estimate_theta", False))
    theta_outer_max = int(getattr(model, "theta_outer_max_iter", 25))
    theta_outer_tol = float(getattr(model, "theta_outer_tol", 1e-5))
    weights = getattr(model, "prior_weights_", None)

    sol = None
    for _outer in range(theta_outer_max if estimate_theta else 1):
        theta_old = float(getattr(family, "theta", 1.0)) if estimate_theta else None

        sol = solve_pirls_gam(
            Z=model.Z,
            y=y,
            penalty_blocks=model.penalty_blocks_,
            smoothing_params=smoothing_params,
            family=family,
            fit_intercept=model.fit_intercept,
            max_iter=int(getattr(model, "max_irls_iter", 100)),
            tol=float(getattr(model, "irls_tol", 1e-8)),
            max_step_halving=int(getattr(model, "max_step_halving", 25)),
            offset=model.offset_train_,
            coef_start=coef_start,
            etastart=etastart,
            mustart=mustart,
        )

        if not estimate_theta:
            break

        mu_hat = np.asarray(sol["mu"], dtype=np.float64)
        theta_new = family.estimate_theta_mle(y, mu_hat, weights=weights)
        family.theta = theta_new

        coef_start = np.asarray(sol["coef_full"], dtype=np.float64).copy()
        if abs(theta_new - theta_old) < theta_outer_tol * (1.0 + abs(theta_old)):
            break

    coef_out = np.asarray(sol["coef_full"], dtype=np.float64).copy()
    model._pirls_last_coef_ = coef_out
    model._pirls_last_eta_ = np.asarray(sol["eta"], dtype=np.float64).copy()
    model._pirls_last_mu_ = np.asarray(sol["mu"], dtype=np.float64).copy()
    if not bool(getattr(model, "_pirls_lock_start_", False)):
        model._pirls_coef_start_ = coef_out.copy()
        model._pirls_eta_start_ = np.asarray(sol["eta"], dtype=np.float64).copy()
        model._pirls_mu_start_ = np.asarray(sol["mu"], dtype=np.float64).copy()
    return FitCoreSolution.from_dict(sol)
