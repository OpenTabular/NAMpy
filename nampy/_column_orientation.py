import numpy as np


def canonical_column_signs(reference, tol=None):
    """
    Deterministic per-column sign choice for orientation-indeterminate bases.

    The sign of each column is chosen so that the first maximum-magnitude entry
    in that column is non-negative. Zero columns keep sign +1.
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("reference must be a 2D matrix.")

    if tol is None:
        scale = float(np.max(np.abs(ref))) if ref.size else 0.0
        tol = float(np.finfo(np.float64).eps * max(1.0, scale) * 32.0)
    else:
        tol = float(tol)

    signs = np.ones(ref.shape[1], dtype=np.float64)
    for j in range(ref.shape[1]):
        col = ref[:, j]
        abs_col = np.abs(col)
        if not np.any(abs_col > tol):
            continue
        idx = int(np.argmax(abs_col))
        if col[idx] < 0.0:
            signs[j] = -1.0
    return signs


def apply_column_signs(matrix, signs):
    arr = np.asarray(matrix, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("matrix must be a 2D matrix.")
    if signs.ndim != 1 or signs.shape[0] != arr.shape[1]:
        raise ValueError("signs must be a vector matching the matrix column count.")
    return np.asarray(arr * signs[np.newaxis, :], dtype=np.float64)
