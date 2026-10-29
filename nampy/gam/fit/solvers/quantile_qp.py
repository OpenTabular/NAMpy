"""Sparse epigraph QP for exact penalized pinball loss.

Variables are p coefficients and n epigraph values. The constraint matrix
has at most 2*n*(p+1) entries; no observation-by-observation dense matrix is
formed. OSQP regularizes its linear solves, not the statistical objective.
"""

import numpy as np
from scipy import sparse


def quantile_problem(X, y, S, weights, level):
    n, p = X.shape
    design = sparse.csc_matrix(X)
    identity = sparse.eye(n, format="csc")
    P = sparse.block_diag(
        (sparse.csc_matrix(S), sparse.csc_matrix((n, n))), format="csc"
    )
    A = sparse.vstack(
        (
            sparse.hstack((level * design, identity)),
            sparse.hstack(((level - 1) * design, identity)),
        ),
        format="csc",
    )
    return P, np.r_[np.zeros(p), weights], A, np.r_[level * y, (level - 1) * y]


def solve_quantile_qp(X, y, S, weights, level, *, max_iter, tol):
    P, q, A, lower = quantile_problem(X, y, S, weights, level)
    if not np.any(S):
        from scipy.optimize import linprog

        result = linprog(
            q,
            A_ub=-A,
            b_ub=-lower,
            bounds=(None, None),
            method="highs",
            options={
                "maxiter": max_iter,
                "primal_feasibility_tolerance": max(tol, 1e-10),
                "dual_feasibility_tolerance": max(tol, 1e-10),
            },
        )
        if result.x is None:
            raise RuntimeError(
                f"Exact quantile LP returned no solution: {result.message}"
            )
        optimum = _kkt_residual(P, q, A, lower, result.x, result.ineqlin.marginals)
        return result.x[: X.shape[1]], result.nit, bool(result.success), optimum

    import osqp

    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=q,
        A=A,
        l=lower,
        u=np.full(len(lower), np.inf),
        eps_abs=tol,
        eps_rel=tol,
        max_iter=max_iter,
        polishing=True,
        adaptive_rho_interval=50,
        verbose=False,
    )
    result = solver.solve(raise_error=False)
    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Exact quantile QP returned no finite solution: {result.info.status}"
        )
    beta = result.x[: X.shape[1]]
    # Include feasibility, stationarity and complementarity, rather than
    # reporting only coefficient stationarity as the previous dense route did.
    optimality = _kkt_residual(P, q, A, lower, result.x, result.y)
    return beta, int(result.info.iter), result.info.status_val == 1, optimality


def _kkt_residual(P, q, A, lower, primal, dual):
    slack = A @ primal - lower
    gradient = P @ primal + q + A.T @ dual
    optimality = float(
        max(
            np.max(np.abs(gradient)),
            np.max(np.maximum(-slack, 0)),
            np.max(np.maximum(dual, 0)),
            np.max(np.abs(dual * slack)),
        )
    )
    return optimality
