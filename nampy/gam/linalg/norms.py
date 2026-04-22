"""Norm helpers matching R/mgcv conventions."""

from __future__ import annotations

import numpy as np


def r_matrix_norm_one(matrix: np.ndarray) -> float:
    """Mirror R matrix ``norm()`` type ``"I"`` convention used by mgcv.

    In mgcv parity-sensitive paths, this is equivalent to the maximum row-sum
    norm (not the column one-norm).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.size == 0:
        return 0.0
    if mat.ndim == 1:
        return float(np.sum(np.abs(mat)))
    return float(np.max(np.sum(np.abs(mat), axis=1)))


def r_matrix_norm_max_abs(matrix: np.ndarray) -> float:
    """Mirror R ``norm(M, "M")`` max-absolute-entry norm."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.size == 0:
        return 0.0
    return float(np.max(np.abs(mat)))


__all__ = ["r_matrix_norm_one", "r_matrix_norm_max_abs"]
