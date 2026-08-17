"""Spectral null-space shrinkage helpers for mgcv-mirror smooths."""

from __future__ import annotations

import numpy as np

from .eigen import symmetric_eigh
from .matrix import symmetrize_matrix


def symmetrize_from_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    """Mirror the lower triangle onto the upper triangle."""
    mat = np.asarray(matrix, dtype=np.float64)
    return np.asarray(np.tril(mat) + np.tril(mat, -1).T, dtype=np.float64)


def geometric_null_space_shrinkage(
    matrix: np.ndarray,
    *,
    shrink: float = 0.1,
    tol: float = 1e-12,
    symmetrize_lower_triangle: bool = False,
    descending: bool = False,
) -> np.ndarray:
    """Replace null eigenvalues by a geometric sequence from the smallest positive one."""
    mat = np.asarray(matrix, dtype=np.float64)
    mat = (
        symmetrize_from_lower_triangle(mat)
        if symmetrize_lower_triangle
        else symmetrize_matrix(mat)
    )

    evals, evecs = symmetric_eigh(
        mat,
        descending=descending,
    )
    tol_eff = float(tol) * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos_mask = evals > tol_eff
    pos = np.asarray(evals[pos_mask], dtype=np.float64)
    if pos.size == 0:
        return mat.copy()

    out = np.asarray(evals, dtype=np.float64).copy()
    null_idx = np.flatnonzero(~pos_mask)
    if null_idx.size:
        base = float(np.min(pos))
        for j, idx in enumerate(null_idx):
            out[int(idx)] = base * (float(shrink) ** (j + 1))
    return np.asarray((evecs * out) @ evecs.T, dtype=np.float64)


def constant_null_space_shrinkage(
    matrix: np.ndarray,
    *,
    shrink: float = 1e-1,
    tol: float = 1e-12,
) -> np.ndarray:
    """Replace all null eigenvalues by one constant multiple of the smallest positive one."""
    mat = symmetrize_matrix(matrix)
    evals, evecs = symmetric_eigh(mat)
    tol_eff = float(tol) * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos_mask = evals > tol_eff
    if not np.any(pos_mask):
        return mat.copy()

    out = np.asarray(evals, dtype=np.float64).copy()
    out[~pos_mask] = float(np.min(evals[pos_mask])) * float(shrink)
    return symmetrize_matrix((evecs * out) @ evecs.T)


__all__ = [
    "symmetrize_from_lower_triangle",
    "geometric_null_space_shrinkage",
    "constant_null_space_shrinkage",
]
