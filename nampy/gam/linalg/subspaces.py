"""Subspace- and spectrum-based invariants for parity checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .eigen import symmetric_eigvalsh
from .rank import numerical_rank


def matrix_self_gram(matrix: np.ndarray) -> np.ndarray:
    """Return ``X X'`` for a 2D matrix."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[1] == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    return np.asarray(mat @ mat.T, dtype=np.float64)


def column_space_projector(matrix: np.ndarray) -> np.ndarray:
    """Return orthogonal projector onto ``col(X)``."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[1] == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    return np.asarray(mat @ np.linalg.pinv(mat), dtype=np.float64)


def row_space_projector(matrix: np.ndarray) -> np.ndarray:
    """Return orthogonal projector onto ``row(X)``."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[0] == 0:
        return np.zeros((mat.shape[1], mat.shape[1]), dtype=np.float64)
    return np.asarray(np.linalg.pinv(mat) @ mat, dtype=np.float64)


def symmetric_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalue spectrum of a symmetric matrix."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square.")
    return np.asarray(np.sort(symmetric_eigvalsh(mat)), dtype=np.float64)


def matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    """Return shape/rank summary for metadata-only parity checks."""
    mat = np.asarray(matrix, dtype=np.float64)
    return {
        "shape": tuple(int(v) for v in mat.shape),
        "rank": int(0 if mat.size == 0 or 0 in mat.shape else numerical_rank(mat)),
    }


def covariance_standard_errors(covariance: np.ndarray) -> np.ndarray:
    """Return marginal standard errors from covariance diagonal."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square.")
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


__all__ = [
    "matrix_self_gram",
    "column_space_projector",
    "row_space_projector",
    "symmetric_spectrum",
    "matrix_summary",
    "covariance_standard_errors",
]
