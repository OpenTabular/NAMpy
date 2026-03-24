import numpy as np
from scipy.linalg import cho_solve

from ..penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
    stabilized_cholesky_solve,
)
from ..covariance import build_bayes_and_freq_covariances
from ..linalg.stacked_qr import (
    gaussian_design_needs_stacked_qr_fit,
    solve_gaussian_penalized_ls_stacked_qr,
)
from ..state import FitCoreSolution


def _prior_weights_vector(model, n: int) -> np.ndarray:
    """Prior observation weights diagonal for Gaussian WLS / deviance / REML."""
    w = getattr(model, "prior_weights_", None)
    if w is None:
        return np.ones(int(n), dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.shape != (int(n),):
        raise ValueError(
            f"prior_weights must have shape ({n},), got {w.shape}."
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("prior_weights must be finite and non-negative.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("prior_weights must sum to a positive value.")
    return w


def solve_gaussian_fit(model, y, smoothing_params):
    """
    Exact Gaussian penalized (weighted) least-squares solve on the full design.

    Minimises ``(y_work - X beta)' W (y_work - X beta) + beta' S beta`` with
    ``W = diag(prior_weights)``, defaulting to identity.
    At convergence this matches IRLS working weights for the Gaussian identity
    family.

    Offset handling
    ---------------
    The model is:
        eta = offset + X beta
        y   = eta + eps
    so the penalized least-squares solve is applied to y - offset.
    """
    y = model.family.validate_y(y)
    n = int(y.shape[0])
    w = _prior_weights_vector(model, n)
    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)

    P_full = build_full_penalty_from_blocks(
        penalty_blocks=model.penalty_blocks_,
        smoothing_params=smoothing_params,
        fit_intercept=model.fit_intercept,
        n_coef=model.n_coef_,
    )

    y_work = y if model.offset_train_ is None else (y - model.offset_train_)

    XtWX = X.T @ (w[:, np.newaxis] * X)
    Xtwy = X.T @ (w * y_work)
    A = XtWX + P_full

    if gaussian_design_needs_stacked_qr_fit(model):
        pls = solve_gaussian_penalized_ls_stacked_qr(
            X,
            y_work,
            w,
            P_full,
            penalty_blocks=model.penalty_blocks_,
            fit_intercept=model.fit_intercept,
            n_coef=model.n_coef_,
        )
        beta_full = np.asarray(pls["coef_full"], dtype=np.float64).ravel()
        eta = X @ beta_full if model.offset_train_ is None else model.offset_train_ + X @ beta_full
        resid = y - eta
        wrss = float(np.sum(w * resid * resid))
        penalty_quadratic = float(pls["penalty_quadratic"])
        H_coef = np.asarray(pls["coef_hat_matrix"], dtype=np.float64)
        trace_H = float(np.trace(H_coef))
        edf = trace_H
        chol_inv = np.asarray(
            pls["XtWX_plus_penalty_chol_inverse_embedded"], dtype=np.float64
        )
        denom = float(model.n_samples_ - edf)
        if not np.isfinite(denom) or denom <= 0.0:
            denom = np.finfo(np.float64).eps
        scale = wrss / denom
        Vp = scale * (chol_inv @ chol_inv.T)
        Vf = scale * (H_coef @ (chol_inv @ chol_inv.T))
        A_inv = np.linalg.pinv(A, hermitian=True, rcond=1e-12)
        return FitCoreSolution(
            coef_full=beta_full.copy(),
            intercept=float(beta_full[0]) if model.fit_intercept else 0.0,
            beta=beta_full[1:].copy() if model.fit_intercept else beta_full.copy(),
            eta=eta,
            mu=eta,
            rss=wrss,
            deviance=wrss,
            edf=edf,
            trace_H=trace_H,
            scale=float(scale),
            cov_bayes=Vp,
            cov_freq=Vf,
            H_coef=H_coef,
            X=X,
            A=A,
            A_inv=A_inv,
            XtWX=XtWX,
            P=P_full,
            penalty_matrix=P_full,
            working_weights=np.asarray(w, dtype=np.float64).copy(),
            working_response=y_work.copy(),
            penalty_quadratic=penalty_quadratic,
            loglik=float(
                np.sum(
                    w
                    * np.asarray(
                        model.family.loglik_obs(y, eta, scale=scale),
                        dtype=np.float64,
                    )
                )
            ),
            offset=None if model.offset_train_ is None else model.offset_train_.copy(),
            log_det_XtWX_plus_penalty=float(pls["log_det_XtWX_plus_penalty"]),
        )

    beta_full, cA, loA, _ = stabilized_cholesky_solve(A, Xtwy)
    eta = X @ beta_full if model.offset_train_ is None else model.offset_train_ + X @ beta_full
    resid = y - eta
    wrss = float(np.sum(w * resid * resid))
    penalty_quadratic = float(beta_full @ (P_full @ beta_full))

    A_inv = cho_solve((cA, loA), np.eye(A.shape[0]), check_finite=False)
    trace_H = float(np.trace(A_inv @ XtWX))
    edf = trace_H

    denom = float(model.n_samples_ - edf)
    if not np.isfinite(denom) or denom <= 0.0:
        denom = np.finfo(np.float64).eps
    scale = wrss / denom
    Vp, Vf, H_coef = build_bayes_and_freq_covariances(scale, A_inv, XtWX)

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
        rss=wrss,
        deviance=wrss,
        edf=edf,
        trace_H=trace_H,
        scale=scale,
        cov_bayes=Vp,
        cov_freq=Vf,
        H_coef=H_coef,
        X=X,
        A=A,
        A_inv=A_inv,
        XtWX=XtWX,
        P=P_full,
        penalty_matrix=P_full,
        working_weights=np.asarray(w, dtype=np.float64).copy(),
        working_response=y_work.copy(),
        penalty_quadratic=penalty_quadratic,
        loglik=float(
            np.sum(
                w
                * np.asarray(
                    model.family.loglik_obs(y, eta, scale=scale),
                    dtype=np.float64,
                )
            )
        ),
        offset=None if model.offset_train_ is None else model.offset_train_.copy(),
    )
