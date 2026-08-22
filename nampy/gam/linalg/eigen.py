"""Shared symmetric-eigensystem and PSD-root helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .matrix import symmetrize_matrix


def symmetric_eigh(
    matrix: np.ndarray,
    *,
    descending: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of a symmetric matrix after explicit symmetrization."""
    mat = symmetrize_matrix(matrix)
    evals, evecs = np.linalg.eigh(mat)
    if descending and evals.size:
        order = np.argsort(evals)[::-1]
        evals = np.asarray(evals[order], dtype=np.float64)
        evecs = np.asarray(evecs[:, order], dtype=np.float64)
    else:
        evals = np.asarray(evals, dtype=np.float64)
        evecs = np.asarray(evecs, dtype=np.float64)
    return evals, evecs


def symmetric_eigvalsh(
    matrix: np.ndarray,
) -> np.ndarray:
    """Eigenvalues of a symmetric matrix after explicit symmetrization."""
    mat = symmetrize_matrix(matrix)
    return np.asarray(np.linalg.eigvalsh(mat), dtype=np.float64)


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetric square root of a PSD matrix."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    evals, evecs = symmetric_eigh(mat)
    evals = np.clip(evals, 0.0, None)
    return np.asarray(evecs @ np.diag(np.sqrt(evals)), dtype=np.float64)


def positive_semidefinite_root(
    matrix: np.ndarray,
    *,
    rank: int | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return leading PSD root columns ``R`` with ``R R'`` on kept eigenspace."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("PSD root requires a square matrix.")
    if mat.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)

    evals, evecs = symmetric_eigh(mat, descending=True)
    tol_eff = float(tol) * max(1.0, float(np.max(np.abs(evals))))
    keep = np.flatnonzero(evals > tol_eff)
    if rank is not None and int(rank) >= 0:
        keep = keep[: min(int(rank), keep.size)]
    if keep.size == 0:
        return np.empty((mat.shape[0], 0), dtype=np.float64)
    return np.asarray(evecs[:, keep] * np.sqrt(evals[keep])[np.newaxis, :], dtype=np.float64)


def symmetric_eigen_partition(
    matrix: np.ndarray,
    *,
    tol: float = 1e-10,
    descending: bool = False,
) -> dict[str, Any]:
    """Partition symmetric eigensystem into null and positive spaces."""
    evals, evecs = symmetric_eigh(matrix, descending=descending)
    scale = float(np.max(np.abs(evals))) if evals.size else 1.0
    tol_eff = float(tol) * max(1.0, scale)
    null_mask = evals <= tol_eff
    pos_mask = ~null_mask
    return {
        "evals": evals,
        "U": evecs,
        "U0": evecs[:, null_mask],
        "U1": evecs[:, pos_mask],
        "d_pos": evals[pos_mask],
        "null_mask": null_mask,
        "pos_mask": pos_mask,
        "rank": int(np.sum(pos_mask)),
        "null_space_dim": int(np.sum(null_mask)),
        "tol_eff": tol_eff,
    }


__all__ = [
    "symmetric_eigh",
    "symmetric_eigvalsh",
    "matrix_sqrt_psd",
    "positive_semidefinite_root",
    "symmetric_eigen_partition",
]
