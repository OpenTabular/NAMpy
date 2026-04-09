import numpy as np

from ..linalg.stacked_qr import (
    balanced_penalty_template_sqrt_for_rank,
    gaussian_design_needs_stacked_qr_fit,
)
from ..penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from ..state import FitCoreSolution
from .irls_core import irls_core


def _prior_weights_vector(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(n), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (int(n),):
        raise ValueError(f"prior_weights must have shape ({n},), got {w.shape}.")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("prior_weights must be finite and non-negative.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("prior_weights must sum to a positive value.")
    return w


def solve_gaussian_fit(model, y, smoothing_params, weights=None):
    y = model.family.validate_y(y)
    n = int(y.shape[0])
    w = _prior_weights_vector(weights, n)
    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)
    S = build_full_penalty_from_blocks(
        penalty_blocks=model.penalty_blocks_,
        smoothing_params=smoothing_params,
        fit_intercept=model.fit_intercept,
        n_coef=model.n_coef_,
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        model.penalty_blocks_,
        fit_intercept=model.fit_intercept,
        n_coef=int(model.n_coef_),
    )

    sol = irls_core(
        X,
        y,
        model.family,
        S,
        offset=model.offset_train_,
        weights=w,
        fit_intercept=model.fit_intercept,
        max_iter=1,
        tol=float(getattr(model, "irls_tol", 1e-8)),
        max_step_halving=int(getattr(model, "max_step_halving", 25)),
        penalty_rank_rows=rank_rows,
        force_stacked_qr=bool(getattr(model, "_use_stacked_qr", False))
        or gaussian_design_needs_stacked_qr_fit(model),
    )

    eta = np.asarray(sol["eta"], dtype=np.float64)
    resid = y - eta
    wrss = float(np.sum(w * resid * resid))
    scale = model.family.estimate_dispersion(y, eta, edf=sol["trace_H"], weights=w)
    sol["rss"] = wrss
    sol["deviance"] = wrss
    sol["scale"] = scale
    sol["working_weights"] = w.copy()
    sol["fisher_weights"] = w.copy()
    sol["working_response"] = (
        y.copy() if model.offset_train_ is None else (y - model.offset_train_).copy()
    )
    sol["loglik"] = float(
        np.sum(
            w
            * np.asarray(
                model.family.loglik_obs(y, eta, scale=scale),
                dtype=np.float64,
            )
        )
    )

    return FitCoreSolution.from_dict(sol)
