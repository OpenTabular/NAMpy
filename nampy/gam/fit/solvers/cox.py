"""Fixed-penalty Cox coefficient solving; no automatic smoothing fallback."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ...losses.cox import cox_partial_likelihood


@dataclass(frozen=True)
class CoxSolution:
    coefficients: np.ndarray
    covariance: np.ndarray
    partial_log_likelihood: float
    iterations: int


def fit_cox(matrix, penalty, time, event, strata, offset, *, max_iter, tol):
    """Newton-CG with exact risk-set gradient/Hessian in identified coordinates.

    This is a fixed-smoothing solver with validated mgcv endpoints. It does not
    claim mgcv's outer REML iteration or smoothing derivatives. Nonconvergence,
    nonfinite curvature and singularity propagate without solver rescue.
    """

    def objective(beta):
        value, gradient, _ = cox_partial_likelihood(
            beta, matrix, time, event, strata, offset
        )
        return -value + 0.5 * beta @ penalty @ beta, -gradient + penalty @ beta

    def hessian(beta):
        return (
            -cox_partial_likelihood(beta, matrix, time, event, strata, offset)[2]
            + penalty
        )

    solved = minimize(
        objective,
        np.zeros(matrix.shape[1]),
        jac=True,
        hess=hessian,
        method="Newton-CG",
        options={"xtol": tol, "maxiter": max_iter},
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise RuntimeError(
            f"Fixed-smoothing Cox coefficient fitting failed: {solved.message}"
        )
    curvature = hessian(solved.x)
    if not np.isfinite(curvature).all() or not np.isfinite(solved.fun):
        raise RuntimeError("Cox coefficient curvature/likelihood is nonfinite.")
    np.linalg.cholesky(curvature)
    covariance = np.linalg.solve(curvature, np.eye(len(solved.x)))
    likelihood = -float(solved.fun) + 0.5 * solved.x @ penalty @ solved.x
    return CoxSolution(solved.x.copy(), covariance, float(likelihood), int(solved.nit))
