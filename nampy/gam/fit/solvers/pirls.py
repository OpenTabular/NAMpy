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
    )


def solve_pirls_fit(model, y, smoothing_params):
    """
    Model-level entry point for the penalized IRLS solver.

    Extracts the design matrix, penalty blocks, and offset from the model,
    runs :func:`fit_pirls_core`, and stores the converged coefficient vector
    on the model for use as a warm start in subsequent iterations.

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
        coef_start=coef_start,
    )
    coef_out = np.asarray(sol["coef_full"], dtype=np.float64).copy()
    model._pirls_last_coef_ = coef_out
    if not bool(getattr(model, "_pirls_lock_start_", False)):
        model._pirls_coef_start_ = coef_out.copy()
    return FitCoreSolution.from_dict(sol)
