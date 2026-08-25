"""QR helpers that mirror parity-sensitive ``mgcv`` representations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def r_linpack_qr_no_pivot(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror base R's ``dqrdc2`` Householder path for ``tol=0``.

    ``mgcv::testStat`` calls ``qr(X, tol=0)``.  In R's modified LINPACK
    ``dqrdc2`` routine, the limited pivot condition is
    ``reduced_norm < original_norm * tol``; hence ``tol=0`` preserves natural
    column order.  Keeping the Householder arithmetic here avoids substituting
    a LAPACK QR whose legal but different rounding can materially change an
    ill-conditioned test statistic.
    """

    packed = np.asarray(matrix, dtype=np.float64, order="F").copy(order="F")
    n_rows, n_cols = packed.shape
    n_reflectors = min(int(n_rows), int(n_cols))
    qraux = np.zeros(n_reflectors, dtype=np.float64)
    for j in range(n_reflectors):
        column = packed[j:, j]
        norm = float(np.linalg.norm(column))
        if norm == 0.0:
            qraux[j] = 0.0
            continue
        if packed[j, j] != 0.0:
            norm = float(np.copysign(norm, packed[j, j]))
        packed[j:, j] = packed[j:, j] / norm
        packed[j, j] = 1.0 + packed[j, j]
        qraux[j] = packed[j, j]
        if j + 1 < n_cols:
            reflector = packed[j:, j]
            denominator = float(reflector[0])
            for col in range(j + 1, n_cols):
                step = -float(np.dot(reflector, packed[j:, col])) / denominator
                packed[j:, col] = packed[j:, col] + step * reflector
        packed[j, j] = -norm
    return packed, qraux


def r_linpack_qr_r(packed_qr: np.ndarray) -> np.ndarray:
    """Return ``qr.R`` from :func:`r_linpack_qr_no_pivot`."""

    packed_qr = np.asarray(packed_qr, dtype=np.float64)
    n_rows, n_cols = packed_qr.shape
    n_triangular_rows = min(int(n_rows), int(n_cols))
    result = np.triu(packed_qr[:n_triangular_rows, :n_cols])
    if n_triangular_rows < n_cols:
        result = np.vstack(
            [
                result,
                np.zeros(
                    (n_cols - n_triangular_rows, n_cols), dtype=np.float64
                ),
            ]
        )
    return np.asarray(result[:n_cols, :n_cols], dtype=np.float64)


def r_linpack_qy(
    packed_qr: np.ndarray, qraux: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Apply ``Q`` from the base-R Householder representation to ``values``."""

    packed_qr = np.asarray(packed_qr, dtype=np.float64)
    qraux = np.asarray(qraux, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    n_reflectors = int(min(qraux.size, packed_qr.shape[1]))
    for j in range(n_reflectors - 1, -1, -1):
        if qraux[j] == 0.0:
            continue
        reflector = packed_qr[j:, j].copy()
        reflector[0] = qraux[j]
        denominator = float(reflector[0])
        for col in range(out.shape[1]):
            step = -float(np.dot(reflector, out[j:, col])) / denominator
            out[j:, col] = out[j:, col] + step * reflector
    return np.asarray(out, dtype=np.float64)


def r_linpack_qty(
    packed_qr: np.ndarray, qraux: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Apply ``t(Q)`` from the base-R LINPACK Householder representation."""
    packed_qr = np.asarray(packed_qr, dtype=np.float64)
    qraux = np.asarray(qraux, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    for index in range(min(qraux.size, packed_qr.shape[1])):
        if qraux[index] == 0.0:
            continue
        reflector = packed_qr[index:, index].copy()
        reflector[0] = qraux[index]
        denominator = float(reflector[0])
        for column in range(out.shape[1]):
            step = -float(np.dot(reflector, out[index:, column])) / denominator
            out[index:, column] += step * reflector
    return np.asarray(out, dtype=np.float64)


def mgcv_pqr_r(matrix: np.ndarray) -> np.ndarray:
    """Mirror ``mgcv:::pqr.R(pqr(matrix))`` in natural column order.

    The packed source used by ``mgcv/src/mat.c::getRpqr()`` has the original
    row stride.  In particular, a wide input still produces a square ``p`` by
    ``p`` result, so padding an economy QR with zero rows is not equivalent.
    For rank-deficient wide inputs this raw factor is not an identified object:
    upstream-derived EDF2/Wald values can change under a pure row permutation.
    Tests must therefore compare only downstream quantities that remain stable.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    _q, r_pivoted, pivot = qr(
        matrix,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    r_pivoted = np.asarray(r_pivoted, dtype=np.float64)
    packed_rows, n_cols = r_pivoted.shape
    packed = np.concatenate(
        [
            np.asarray(r_pivoted, dtype=np.float64, order="F").ravel(order="F"),
            np.zeros(n_cols * n_cols, dtype=np.float64),
        ]
    )
    r_square = np.zeros((n_cols, n_cols), dtype=np.float64)
    for j in range(n_cols):
        stop = j + 1
        base = packed_rows * j
        r_square[:stop, j] = packed[base : base + stop]

    r_natural = np.zeros_like(r_square)
    r_natural[:, np.asarray(pivot, dtype=np.intp)] = r_square
    return r_natural


__all__ = [
    "mgcv_pqr_r",
    "r_linpack_qr_no_pivot",
    "r_linpack_qr_r",
    "r_linpack_qty",
    "r_linpack_qy",
]
