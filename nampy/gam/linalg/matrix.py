"""Dense matrix utility helpers."""

from __future__ import annotations

import numpy as np


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (A + A')`` as float64."""
    mat = np.asarray(matrix, dtype=np.float64)
    return np.asarray(0.5 * (mat + mat.T), dtype=np.float64)


__all__ = ["symmetrize_matrix"]
