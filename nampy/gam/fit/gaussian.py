import numpy as np
from scipy.linalg import cho_solve

from .penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
    stabilized_cholesky_solve,
)
from .covariance import build_bayes_and_freq_covariances
from .state import FitCoreSolution


def solve_gaussian_fit(model, y, smoothing_params):
    """
    Exact Gaussian penalized least-squares solve on the full design.

    Offset handling
    ---------------
    The model is:
        eta = offset + X beta
        y   = eta + eps
    so the penalized least-squares solve is applied to y - offset.
    """
    y = model.family.validate_y(y)
    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)

    P_full = build_full_penalty_from_blocks(
        penalty_blocks=model.penalty_blocks_,
        smoothing_params=smoothing_params,
        fit_intercept=model.fit_intercept,
        n_coef=model.n_coef_,
    )

    y_work = y if model.offset_train_ is None else (y - model.offset_train_)

    XtX = X.T @ X
    Xty = X.T @ y_work
    A = XtX + P_full

    beta_full, cA, loA, _ = stabilized_cholesky_solve(A, Xty)
    eta = X @ beta_full if model.offset_train_ is None else model.offset_train_ + X @ beta_full
    resid = y - eta
    rss = float(resid @ resid)
    penalty_quadratic = float(beta_full @ (P_full @ beta_full))

    A_inv = cho_solve((cA, loA), np.eye(A.shape[0]), check_finite=False)
    trace_H = float(np.trace(A_inv @ XtX))
    edf = trace_H

    scale = rss / max(model.n_samples_ - edf, 1.0)
    Vp, Vf, H_coef = build_bayes_and_freq_covariances(scale, A_inv, XtX)

    if model.fit_intercept:
        intercept = float(beta_full[0])
        beta_term = beta_full[1:].copy()
    else:
        intercept = 0.0
        beta_term = beta_full.copy()

    return FitCoreSolution(
        coef_full=beta_full.copy(),
        intercept=intercept,
        beta=beta_term,
        eta=eta,
        mu=eta,
        rss=rss,
        deviance=rss,
        edf=edf,
        trace_H=trace_H,
        scale=scale,
        cov_bayes=Vp,
        cov_freq=Vf,
        H_coef=H_coef,
        X=X,
        A=A,
        A_inv=A_inv,
        XtWX=XtX,
        P=P_full,
        penalty_matrix=P_full,
        working_weights=np.ones(X.shape[0], dtype=np.float64),
        working_response=y_work.copy(),
        penalty_quadratic=penalty_quadratic,
        loglik=float(model.family.loglik(y, eta, scale=scale)),
        offset=None if model.offset_train_ is None else model.offset_train_.copy(),
    )
