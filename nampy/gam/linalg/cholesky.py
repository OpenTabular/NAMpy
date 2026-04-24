"""Pivoted-Cholesky helpers shared by mgcv-mirror solvers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import get_lapack_funcs

from .eigen import symmetric_eigh


def safe_pivoted_cholesky(
    matrix: np.ndarray,
    jitter: np.ndarray,
    *,
    eigen_fix: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Try pivoted Cholesky with mgcv-style ridge inflation and optional eigen fix."""

    def _factor(arr: np.ndarray):
        pstrf = get_lapack_funcs("pstrf", dtype=np.float64)
        chol_arr, piv_arr, rank_found, info = pstrf(
            np.asarray(arr, dtype=np.float64), lower=0
        )
        if int(info) < 0:
            raise np.linalg.LinAlgError(f"dpstrf failed with info={info}.")
        piv0 = np.asarray(piv_arr, dtype=int).ravel() - 1
        ipiv0 = np.empty_like(piv0)
        ipiv0[piv0] = np.arange(piv0.size, dtype=int)
        return (
            np.asarray(np.triu(chol_arr), dtype=np.float64),
            piv0,
            ipiv0,
            int(rank_found) == int(arr.shape[0]) and int(info) == 0,
        )

    last_factor = None

    try:
        chol_arr, piv_arr, ipiv_arr, ok = _factor(matrix)
        if ok:
            return chol_arr, piv_arr, ipiv_arr, True
        last_factor = (chol_arr, piv_arr, ipiv_arr)
    except np.linalg.LinAlgError:
        chol_arr = piv_arr = ipiv_arr = None

    jitter_work = np.asarray(jitter, dtype=np.float64).copy()
    for _ in range(100):
        try:
            chol_arr, piv_arr, ipiv_arr, ok = _factor(matrix + jitter_work)
            if ok:
                return chol_arr, piv_arr, ipiv_arr, False
            last_factor = (chol_arr, piv_arr, ipiv_arr)
        except np.linalg.LinAlgError:
            pass
        jitter_work = jitter_work * 100.0

    if eigen_fix:
        evals, evecs = symmetric_eigh(np.asarray(matrix, dtype=np.float64))
        evals = np.abs(evals)
        eval_max = float(np.max(evals)) if evals.size else 0.0
        if eval_max > 0.0:
            evals = np.where(evals < eval_max * 1e-10, eval_max * 1e-10, evals)
        matrix_fixed = evecs @ (evals[:, None] * evecs.T)
        chol_arr, piv_arr, ipiv_arr, _ = _factor(matrix_fixed)
        return chol_arr, piv_arr, ipiv_arr, False

    if last_factor is not None:
        chol_arr, piv_arr, ipiv_arr = last_factor
        return chol_arr, piv_arr, ipiv_arr, False

    raise np.linalg.LinAlgError("Pivoted Cholesky failed after ridge inflation.")


def chol_solve_pivoted(
    chol_upper: np.ndarray,
    rhs: np.ndarray,
    *,
    piv: np.ndarray | None = None,
    ipiv: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``A x = rhs`` with ``P' A P = R' R`` from pivoted upper Cholesky."""
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if piv is not None:
        rhs_arr = rhs_arr[piv, ...]
    z = solve_triangular(
        chol_upper,
        rhs_arr,
        lower=False,
        trans="T",
        check_finite=False,
    )
    y = solve_triangular(chol_upper, z, lower=False, check_finite=False)
    if ipiv is not None:
        y = y[ipiv, ...]
    return np.asarray(y, dtype=np.float64)


def compute_preconditioned_inverse(
    chol_upper: np.ndarray,
    diagonal_preconditioner: np.ndarray,
    size: int,
    *,
    piv: np.ndarray | None = None,
    ipiv: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``D * (R^{-1} R^{-T}) * D`` for upper-Cholesky ``R`` and diagonal ``D``."""
    eye = np.eye(size, dtype=np.float64)
    sol = chol_solve_pivoted(chol_upper, eye, piv=piv, ipiv=ipiv)
    return diagonal_preconditioner[:, None] * sol * diagonal_preconditioner[None, :]


__all__ = [
    "safe_pivoted_cholesky",
    "chol_solve_pivoted",
    "compute_preconditioned_inverse",
]
