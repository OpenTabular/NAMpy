"""Pivoted-Cholesky helpers shared by mgcv-mirror solvers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .eigen import symmetric_eigh


def pivoted_cholesky(
    matrix: np.ndarray,
    *,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Upper pivoted Cholesky matching the ``DPSTRF`` factor contract.

    The returned permutation satisfies ``A[piv, piv] = R.T @ R`` when the
    factor is full rank. ``tol=None`` uses the upstream ``DPSTRF`` default
    stopping threshold; ``tol=0`` retains every strictly positive pivot.
    """
    source = np.asarray(matrix, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("pivoted Cholesky requires a square matrix.")
    n = int(source.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=int), 0
    if not np.all(np.isfinite(source)):
        raise np.linalg.LinAlgError("pivoted Cholesky requires finite entries.")

    # DPSTRF with UPLO='U' reads the upper triangle of its input.
    upper = np.triu(source)
    work = np.asarray(upper + np.triu(upper, 1).T, dtype=np.float64)
    piv = np.arange(n, dtype=int)
    factor = np.zeros((n, n), dtype=np.float64)
    max_diagonal = float(np.max(np.diag(work)))
    stop = (
        float(n) * np.finfo(np.float64).eps * max_diagonal
        if tol is None
        else float(tol)
    )

    rank = n
    for k in range(n):
        residual_diagonal = np.empty(n - k, dtype=np.float64)
        for offset, column in enumerate(range(k, n)):
            residual_diagonal[offset] = work[column, column] - float(
                factor[:k, column] @ factor[:k, column]
            )
        pivot = k + int(np.argmax(residual_diagonal))
        pivot_value = float(residual_diagonal[pivot - k])
        if not np.isfinite(pivot_value) or pivot_value <= stop:
            rank = k
            break

        if pivot != k:
            work[[k, pivot], :] = work[[pivot, k], :]
            work[:, [k, pivot]] = work[:, [pivot, k]]
            piv[[k, pivot]] = piv[[pivot, k]]
            if k:
                factor[:k, [k, pivot]] = factor[:k, [pivot, k]]

        factor[k, k] = np.sqrt(pivot_value)
        for column in range(k + 1, n):
            numerator = work[k, column] - float(
                factor[:k, k] @ factor[:k, column]
            )
            factor[k, column] = numerator / factor[k, k]

    return np.asarray(factor, dtype=np.float64), piv, int(rank)


def mgcv_mroot_chol(
    matrix: np.ndarray,
    *,
    rank: int | None = None,
) -> np.ndarray:
    """Port ``mgcv::mroot(..., method="chol")`` as ``B`` with ``B B' = A``.

    The particular root is not identified and must not be compared directly.
    This representation is nevertheless needed internally where upstream feeds
    the root into a subsequent QR factorization, as in ``mgcv::recov``.
    """

    source = np.asarray(matrix, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("mroot requires a square matrix.")
    n = int(source.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    factor, pivot, rank_found = pivoted_cholesky(
        0.5 * (source + source.T), tol=0.0
    )
    if rank_found < n:
        factor[rank_found:, rank_found:] = 0.0

    factor = factor[:, np.argsort(np.asarray(pivot, dtype=np.int64))]
    rank_use = (
        int(rank_found)
        if rank is None or int(rank) < 1
        else min(int(rank), n)
    )
    if rank_use == 0:
        return np.empty((n, 0), dtype=np.float64)
    return np.asarray(factor[:rank_use, :].T, dtype=np.float64)


def safe_pivoted_cholesky(
    matrix: np.ndarray,
    jitter: np.ndarray,
    *,
    eigen_fix: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Port the pivoted-Cholesky/ridge loop in ``mgcv::gam.fit5``."""
    matrix_work = np.asarray(matrix, dtype=np.float64).copy()
    n = int(matrix_work.shape[0])
    chol_arr, piv_arr, rank_found = pivoted_cholesky(matrix_work)
    initially_full_rank = rank_found == n
    jitter_work = np.asarray(jitter, dtype=np.float64).copy()
    multiplier = 1.0

    while rank_found < n:
        if eigen_fix:
            evals, evecs = symmetric_eigh(matrix_work)
            positive = evals[evals > 0.0]
            if positive.size == 0:
                raise np.linalg.LinAlgError(
                    "Cannot apply mgcv eigen fix without a positive eigenvalue."
                )
            threshold = max(
                float(np.min(positive)),
                float(np.max(evals)) * 1e-6,
            ) * multiplier
            multiplier *= 10.0
            evals = np.where(evals < threshold, threshold, evals)
            matrix_work = np.asarray(
                evecs @ (evals[:, np.newaxis] * evecs.T),
                dtype=np.float64,
            )
            chol_arr, piv_arr, rank_found = pivoted_cholesky(matrix_work)
        else:
            chol_arr, piv_arr, rank_found = pivoted_cholesky(
                matrix_work + jitter_work
            )
            jitter_work *= 100.0
            if not np.all(np.isfinite(jitter_work)):
                raise np.linalg.LinAlgError(
                    "mgcv ridge inflation overflowed before Cholesky became full rank."
                )

    ipiv_arr = np.empty_like(piv_arr)
    ipiv_arr[piv_arr] = np.arange(piv_arr.size, dtype=int)
    return chol_arr, piv_arr, ipiv_arr, bool(initially_full_rank)


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
    return np.asarray(
        diagonal_preconditioner[:, None]
        * sol
        * diagonal_preconditioner[None, :],
        dtype=np.float64,
    )


__all__ = [
    "mgcv_mroot_chol",
    "pivoted_cholesky",
    "safe_pivoted_cholesky",
    "chol_solve_pivoted",
    "compute_preconditioned_inverse",
]
