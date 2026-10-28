"""Convex, fixed-penalty expectile and quantile regression.

Penalties use ``0.5 * beta.T @ S @ beta``; observation losses are sums,
never means. No likelihood or Gaussian uncertainty is attached to these fits.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class LossFit:
    """Numerical solution: coefficients (p,), fitted values (n,), diagnostics."""

    coefficients: np.ndarray
    fitted: np.ndarray
    objective: float
    iterations: int
    converged: bool
    optimality: float


def _expectile_loss(residual, level):
    """Per-observation asymmetric squared loss used for estimator scoring."""
    return np.where(residual >= 0, level, 1 - level) * residual**2


def _pinball_loss(residual, level):
    """Exact quantile scoring loss, including when training uses smooth ELF."""
    return np.maximum(level * residual, (level - 1) * residual)


def _inputs(design, response, penalty, weights, level):
    X = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],) or not X.size:
        raise ValueError("design must be (n,p) and response must be (n,).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("design and response must be finite.")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be strictly between zero and one.")
    S = (
        np.zeros((X.shape[1], X.shape[1]))
        if penalty is None
        else np.asarray(penalty, dtype=float)
    )
    if S.shape != (X.shape[1], X.shape[1]) or not np.all(np.isfinite(S)):
        raise ValueError("penalty must be a finite (p,p) matrix.")
    if not np.allclose(S, S.T, rtol=1e-12, atol=1e-12):
        raise ValueError("penalty must be symmetric.")
    S = (S + S.T) / 2
    if np.linalg.eigvalsh(S).min() < -1e-10 * max(1, np.linalg.norm(S, 2)):
        raise ValueError("penalty must be positive semidefinite.")
    w = np.ones(y.size) if weights is None else np.asarray(weights, dtype=float)
    if (
        w.shape != y.shape
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or not np.any(w > 0)
    ):
        raise ValueError("weights must be finite, nonnegative (n,), with positive sum.")
    return X, y, S, w


def _controls(max_iter, tol):
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    return int(max_iter)


def fit_expectile(
    design, response, *, level=0.5, penalty=None, weights=None, max_iter=200, tol=1e-9
):
    """Minimize weighted asymmetric squared error plus ``b'Sb/2``.

    Parameters
    ----------
    design : array-like of float, shape (n, p)
        Identifiable basis, including any intercept column.
    response : array-like of float, shape (n,)
        Finite response after subtracting known offsets.
    level : float, default 0.5
        Expectile asymmetry in (0,1). At 0.5 this is penalized least squares.
    penalty : array-like of float, shape (p,p), optional
        Complete positive semidefinite penalty, already scaled by smoothing.
    weights : array-like of float, shape (n,), optional
        Nonnegative observation weights; defaults to ones.
    max_iter : int, default 200
        Maximum LAWS solves.
    tol : float, default 1e-9
        Relative first-order convergence tolerance.

    Returns
    -------
    LossFit
        Coefficients (p,), predictions (n,), penalized objective, iteration
        count, convergence flag, and infinity norm of the objective gradient.
    """
    max_iter = _controls(max_iter, tol)
    X, y, S, w = _inputs(design, response, penalty, weights, level)
    beta = lstsq(X.T @ (w[:, None] * X) + S, X.T @ (w * y))[0]
    converged = False
    for _iteration in range(1, max_iter + 1):
        residual = y - X @ beta
        asym = np.where(residual >= 0, level, 1 - level)
        v = 2 * w * asym
        candidate = lstsq(X.T @ (v[:, None] * X) + S, X.T @ (v * y))[0]

        # LAWS can overshoot when many residual signs change. Backtracking
        # keeps the convex objective decreasing without altering its target.
        def objective(b):
            r = y - X @ b
            return (
                np.sum(w * np.where(r >= 0, level, 1 - level) * r**2) + 0.5 * b @ S @ b
            )

        old = objective(beta)
        for _ in range(50):
            if objective(candidate) <= old + 1e-12 * (1 + abs(old)):
                break
            candidate = (candidate + beta) / 2
        beta = candidate
        residual = y - X @ beta
        v = 2 * w * np.where(residual >= 0, level, 1 - level)
        gradient = S @ beta - X.T @ (v * residual)
        optimality = float(np.max(np.abs(gradient)))
        if optimality <= tol * (1 + np.linalg.norm(X.T @ (w * y), np.inf)):
            converged = True
            break
    return LossFit(
        beta, X @ beta, float(objective(beta)), _iteration, converged, optimality
    )


def fit_quantile(
    design,
    response,
    *,
    level=0.5,
    penalty=None,
    weights=None,
    bandwidth=0.0,
    max_iter=10000,
    tol=1e-9,
):
    """Minimize weighted pinball (or smooth ELF) loss plus b'Sb/2.

    Inputs have the same shapes and semantics as :func:`fit_expectile`.
    ``level`` defaults to 0.5; ``max_iter=10000`` and ``tol=1e-9`` control
    optimization. ``bandwidth`` is a nonnegative response-scale float,
    default 0: zero uses a sparse exact convex epigraph QP; positive values
    use ``(level-1)*r + bandwidth*log(1+exp(r/bandwidth))``. This is qgam's
    fixed-learning-rate ELF objective (sigma=1, co=bandwidth), up to a
    coefficient-independent constant. It does not calibrate qgam inference.

    Returns a :class:`LossFit`; for exact pinball, ``optimality`` measures
    the maximum KKT feasibility, stationarity and complementarity residual.
    A convergence flag never implies calibration. The sparse constraint storage
    grows with n*p rather than n**2; solver work and convergence still depend
    on the data and conditioning.
    """
    max_iter = _controls(max_iter, tol)
    X, y, S, w = _inputs(design, response, penalty, weights, level)
    if not np.isfinite(bandwidth) or bandwidth < 0:
        raise ValueError("bandwidth must be finite and nonnegative.")
    n, p = X.shape
    beta = lstsq(X, y)[0]
    if bandwidth > 0:

        def fun(b):
            r = y - X @ b
            value = (
                w @ ((level - 1) * r + bandwidth * np.logaddexp(0, r / bandwidth))
                + 0.5 * b @ S @ b
            )
            gradient = S @ b - X.T @ (w * (level - 1 + expit(r / bandwidth)))
            return value, gradient

        def hess(b):
            prob = expit((y - X @ b) / bandwidth)
            return S + X.T @ ((w * prob * (1 - prob) / bandwidth)[:, None] * X)

        result = minimize(
            fun,
            beta,
            jac=True,
            hess=hess,
            method="trust-exact",
            options={"gtol": tol, "maxiter": max_iter},
        )
        optimality = float(np.max(np.abs(fun(result.x)[1])))
        converged = optimality <= max(tol * 10, 1e-7)
        return LossFit(
            result.x, X @ result.x, float(result.fun), result.nit, converged, optimality
        )
    from ._quantile_qp import solve_quantile_qp

    beta, iterations, converged, optimality = solve_quantile_qp(
        X, y, S, w, level, max_iter=max_iter, tol=tol
    )
    residual = y - X @ beta
    objective = float(
        w @ np.maximum(level * residual, (level - 1) * residual) + 0.5 * beta @ S @ beta
    )
    return LossFit(beta, X @ beta, objective, iterations, converged, optimality)
