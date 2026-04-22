"""Shared numerical-rank and null-space helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from .eigen import symmetric_eigh, symmetric_eigvalsh


def numerical_rank(matrix: np.ndarray, *, hermitian: bool = False) -> int:
    """Float64 numerical rank wrapper used across GAM internals."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("numerical_rank requires a 2D matrix.")
    if 0 in mat.shape:
        return 0
    if hermitian:
        evals = symmetric_eigvalsh(mat)
        scale = float(np.max(np.abs(evals))) if evals.size else 0.0
        tol = np.finfo(np.float64).eps * max(mat.shape) * max(scale, 1.0)
        return int(np.sum(np.abs(evals) > tol))
    return int(np.linalg.matrix_rank(mat))


def matrix_is_rank_deficient(matrix: np.ndarray) -> bool:
    """Whether a 2D matrix has rank smaller than its column count."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix_is_rank_deficient requires a 2D matrix.")
    return numerical_rank(mat) < int(mat.shape[1])


def svd_null_space_basis(
    matrix: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Right null-space basis from SVD plus detected rank."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("svd_null_space_basis requires a 2D matrix.")
    _, q = mat.shape
    if q == 0:
        return np.empty((0, 0), dtype=np.float64), 0

    _, s, vt = np.linalg.svd(mat, full_matrices=True)
    if s.size == 0:
        return np.empty((q, 0), dtype=np.float64), 0

    smax = float(s[0])
    tol = max(smax * float(sv_rel_tol), np.finfo(np.float64).eps * max(smax, 1.0))
    rank_x = int(np.sum(s > tol))
    if rank_x >= q:
        return np.empty((q, 0), dtype=np.float64), rank_x
    return np.asarray(vt[rank_x:q, :], dtype=np.float64).T, rank_x


def project_coef_onto_row_space(
    X: np.ndarray,
    coef_full: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> np.ndarray:
    """Orthogonally remove the ``null(X)`` component from a coefficient vector."""
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(coef_full, dtype=np.float64).ravel()
    q = int(X.shape[1])
    if b.shape[0] != q:
        raise ValueError(f"coef_full has length {b.shape[0]}, expected {q}.")
    if q == 0:
        return b.copy()
    null_basis, _rank_x = svd_null_space_basis(X, sv_rel_tol=sv_rel_tol)
    if null_basis.shape[1] == 0:
        return b.copy()
    return np.asarray(b - null_basis @ (null_basis.T @ b), dtype=np.float64)


def snap_coef_to_reference_null_space(
    coef_full: np.ndarray,
    X: np.ndarray,
    coef_reference: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> np.ndarray:
    """Keep fitted values fixed while copying null-space coordinates from a reference."""
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(coef_full, dtype=np.float64).ravel()
    ref = np.asarray(coef_reference, dtype=np.float64).ravel()
    _, q = X.shape
    if b.shape[0] != q or ref.shape[0] != q:
        raise ValueError("coef vectors must have length q matching X.shape[1].")
    if q == 0:
        return b.copy()
    null_basis, _rank_x = svd_null_space_basis(X, sv_rel_tol=sv_rel_tol)
    if null_basis.shape[1] == 0:
        return b.copy()
    projector = np.asarray(null_basis @ null_basis.T, dtype=np.float64)
    return np.asarray(b + projector @ (ref - b), dtype=np.float64)


def balanced_penalty_template_sqrt_for_rank(
    penalty_blocks: Iterable[Any],
    *,
    fit_intercept: bool,
    n_coef: int,
) -> np.ndarray:
    """Aggregate penalty-template square root used only for stacked-QR rank detection."""
    offset = 1 if fit_intercept else 0
    q = offset + int(n_coef)
    template_sum = np.zeros((q, q), dtype=np.float64)
    for pb in penalty_blocks:
        smooth_idx = int(getattr(pb, "smoothing_index", -1))
        if smooth_idx < 0:
            continue
        template_block = np.asarray(pb.matrix, dtype=np.float64)
        sl = pb.coef_slice
        block_dim = int(sl.stop - sl.start)
        if template_block.shape != (block_dim, block_dim):
            continue
        frob = float(np.linalg.norm(template_block, ord="fro"))
        if frob <= 0.0:
            continue
        idx = np.arange(offset + sl.start, offset + sl.stop, dtype=np.int64)
        template_sum[np.ix_(idx, idx)] += template_block / frob
    if q == 0:
        return np.empty((0, 0), dtype=np.float64)
    evals, evecs = symmetric_eigh(template_sum)
    emax = float(np.max(evals)) if evals.size else 0.0
    if emax <= 0.0:
        return np.zeros((0, q), dtype=np.float64)
    thr = emax * (np.finfo(np.float64).eps ** 0.66)
    mask = evals > thr
    if not np.any(mask):
        return np.zeros((0, q), dtype=np.float64)
    vals = np.asarray(evals[mask], dtype=np.float64)
    v_sel = np.asarray(evecs[:, mask], dtype=np.float64)
    sqrt_evals = np.sqrt(np.maximum(vals, 0.0))
    return np.asarray(sqrt_evals[:, np.newaxis] * v_sel.T, dtype=np.float64)


def symmetric_penalty_rank(matrix: np.ndarray, *, tol: float = 1e-10) -> int:
    """Rank of a symmetric penalty matrix using mgcv-style relative thresholding."""
    evals = symmetric_eigvalsh(matrix)
    scale = float(np.max(np.abs(evals))) if evals.size else 1.0
    tol_eff = float(tol) * max(1.0, scale)
    return int(np.sum(evals > tol_eff))


def upper_triangular_rrank(
    matrix: np.ndarray,
    *,
    tol: float | None = None,
) -> int:
    """Mirror ``mgcv::Rrank`` for an upper-triangular factor."""
    R = np.asarray(matrix, dtype=np.float64)
    m = int(R.shape[0])
    rank = min(m, int(R.shape[1]))
    if tol is None:
        tol = float(np.finfo(np.float64).eps ** 0.9)
    trcon = get_lapack_funcs("trcon", (np.asfortranarray(R),))
    while rank > 0:
        block = np.asfortranarray(R[:rank, :rank], dtype=np.float64)
        rcond, info = trcon(block, norm="1", uplo="U", diag="N")
        if info != 0 or not np.isfinite(rcond):
            rcond = 0.0
        if float(rcond) > float(tol):
            break
        rank -= 1
    return int(rank)


__all__ = [
    "numerical_rank",
    "matrix_is_rank_deficient",
    "svd_null_space_basis",
    "project_coef_onto_row_space",
    "snap_coef_to_reference_null_space",
    "balanced_penalty_template_sqrt_for_rank",
    "symmetric_penalty_rank",
    "upper_triangular_rrank",
]
