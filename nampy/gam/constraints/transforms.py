from __future__ import annotations

import numpy as np
from scipy.linalg import blas
from scipy.linalg import qr as scipy_qr


def orthogonal_residual(B, A):
    B = np.asarray(B, dtype=np.float64)
    if A is None or np.asarray(A).size == 0 or np.asarray(A).shape[1] == 0:
        return B
    A = np.asarray(A, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    return B - A @ coef


def dependent_column_indices(
    B,
    A=None,
    *,
    tol: float = 1e-10,
    rank_def: int = 0,
    strict: bool = False,
):
    """
    Mirror mgcv/R/mgcv.r::fixDependence().

    Returns 0-based column indices of ``B`` that should be deleted so that ``B``
    is linearly independent of ``A``.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")
    if B.shape[1] == 0:
        return np.array([], dtype=int)
    if A is None:
        A = np.empty((B.shape[0], 0), dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same row count.")
    if A.shape[1] == 0:
        return np.array([], dtype=int)

    Q1, R1, _piv1 = scipy_qr(A, mode="full", pivoting=True)
    R11 = float(abs(R1[0, 0])) if R1.size else 0.0
    r = int(A.shape[1])
    n = int(A.shape[0])

    if strict:
        QtX2 = Q1.T @ B
        if r < n:
            QtX2[r:, :] = 0.0
        mdiff = np.mean(np.abs(B - Q1 @ QtX2), axis=0)
        if rank_def > 0:
            order = np.argsort(np.argsort(mdiff, kind="mergesort"), kind="mergesort")
            deleted = np.flatnonzero(order < int(rank_def))
        else:
            deleted = np.flatnonzero(mdiff < R11 * float(tol))
        return np.asarray(deleted, dtype=int)

    QtX2 = (Q1.T @ B)[r:n, :]
    if QtX2.size == 0:
        return np.array([], dtype=int)

    _Q2, R2, piv2 = scipy_qr(QtX2, mode="economic", pivoting=True)
    R = np.asarray(R2, dtype=np.float64)
    if R.size == 0:
        return np.array([], dtype=int)

    r_total = int(R.shape[0])
    r0 = r_total
    if 0 < int(rank_def) <= r_total:
        r0 = r_total - int(rank_def)
    else:
        while r0 > 0:
            block = R[r0 - 1 : r_total, r0 - 1 : r_total]
            if np.mean(np.abs(block)) >= R11 * float(tol):
                break
            r0 -= 1
    r0 += 1
    if r0 > r_total:
        return np.array([], dtype=int)
    return np.asarray(piv2[r0 - 1 : r_total], dtype=int)


def independent_column_indices(B, A=None, tol: float = 1e-10):
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")
    if B.shape[1] == 0:
        return np.array([], dtype=int)
    deleted = dependent_column_indices(B, A=A, tol=tol)
    if deleted.size == 0:
        return np.arange(B.shape[1], dtype=int)
    keep_mask = np.ones(B.shape[1], dtype=bool)
    keep_mask[np.asarray(deleted, dtype=int)] = False
    return np.flatnonzero(keep_mask).astype(int, copy=False)


def null_space_basis_from_constraint_matrix(
    C, d: int | None = None, tol: float = 1e-10
):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError("Constraint matrix must be 2D.")
    if d is not None and C.shape[1] != int(d):
        raise ValueError(f"Constraint matrix has width {C.shape[1]}, expected {d}.")
    if C.size == 0:
        return np.eye(C.shape[1], dtype=np.float64), 0
    if C.shape[0] == 1:
        return _single_constraint_null_space_basis_r_qr(C, d=d)
    # Mirror mgcv smoothCon(absorb.cons=TRUE), which forms qr(t(C)) and takes
    # the trailing columns of Q as a constraint null-space basis. An SVD basis
    # spans the same space but can rotate coefficients differently, which shows
    # up in parity-sensitive terms such as t2(full=FALSE).
    Qt, R = scipy_qr(C.T, mode="full", pivoting=False)
    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return np.eye(C.shape[1], dtype=np.float64), 0
    tol_eff = max(
        float(tol),
        float(np.max(diag_R)) * max(C.shape) * np.finfo(float).eps,
    )
    rank = int(np.sum(diag_R > tol_eff))
    return np.asarray(Qt[:, rank:], dtype=np.float64).copy(), rank


def _single_constraint_null_space_basis_r_qr(C, d: int | None = None):
    """
    Mirror base R default ``qr(t(C))`` / ``qr.qty`` for a single dense constraint.

    `mgcv::smoothCon(absorb.cons=TRUE)` uses base R's default `qr`, which for this
    one-column case follows the LINPACK reflector path rather than LAPACK. That
    differs by 1 ulp from `dgeqrf`/SciPy QR on some smooths (notably `bs="mrf"`),
    and those last-bit differences propagate into exact-fit Gaussian residual parity.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != 1:
        raise ValueError("single-constraint QR helper expects a 1 x d matrix.")
    if d is not None and C.shape[1] != int(d):
        raise ValueError(f"Constraint matrix has width {C.shape[1]}, expected {d}.")

    q = int(C.shape[1])
    if q == 0:
        return np.zeros((0, 0), dtype=np.float64), 0
    if q == 1:
        return np.zeros((1, 0), dtype=np.float64), 1

    x = np.asarray(C.T.copy(), dtype=np.float64, order="F")
    qraux = np.zeros(1, dtype=np.float64)

    nrmxl = float(blas.dnrm2(x[:, 0]))
    if nrmxl == 0.0:
        return np.eye(q, dtype=np.float64), 0
    if x[0, 0] != 0.0:
        nrmxl = float(np.copysign(nrmxl, x[0, 0]))
    x[:, 0] = blas.dscal(1.0 / nrmxl, x[:, 0])
    x[0, 0] = 1.0 + x[0, 0]
    qraux[0] = x[0, 0]
    x[0, 0] = -nrmxl

    q_full = np.eye(q, dtype=np.float64, order="F")
    temp = float(x[0, 0])
    x[0, 0] = qraux[0]
    for col in range(q):
        t = -float(blas.ddot(x[:, 0], q_full[:, col])) / float(x[0, 0])
        q_full[:, col] = blas.daxpy(x[:, 0], q_full[:, col], a=t)
    x[0, 0] = temp
    return np.asarray(q_full[:, 1:], dtype=np.float64), 1


def localized_null_space_basis_from_constraint_matrix(
    C, d: int | None = None, tol: float = 1e-10
):
    """
    Compute a null-space basis while preserving coordinates untouched by C.

    When a constraint matrix has support on only a strict subset of coefficient
    coordinates, a generic SVD null-space basis can arbitrarily rotate the full
    coefficient space. That is mathematically valid, but it destroys block-local
    penalty structure such as t2's separate identity penalties. This helper
    keeps inactive coordinates as explicit identity columns and reparameterizes
    only the active block.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError("Constraint matrix must be 2D.")
    if d is not None and C.shape[1] != int(d):
        raise ValueError(f"Constraint matrix has width {C.shape[1]}, expected {d}.")
    if C.size == 0:
        return np.eye(C.shape[1], dtype=np.float64), 0

    active = np.any(np.abs(C) > float(tol), axis=0)
    if not np.any(active):
        return np.eye(C.shape[1], dtype=np.float64), 0
    if np.all(active):
        return null_space_basis_from_constraint_matrix(C, d=d, tol=tol)

    active_idx = np.flatnonzero(active)
    inactive_idx = np.flatnonzero(~active)
    T_active, rank = null_space_basis_from_constraint_matrix(
        C[:, active_idx],
        d=int(active_idx.size),
        tol=tol,
    )
    n_keep_active = int(T_active.shape[1])
    keep_active_idx = active_idx[:n_keep_active]
    drop_active_idx = active_idx[n_keep_active:]
    keep_idx = [
        int(j) for j in range(C.shape[1]) if j not in set(drop_active_idx.tolist())
    ]
    col_pos = {int(j): int(i) for i, j in enumerate(keep_idx)}

    out = np.zeros((C.shape[1], len(keep_idx)), dtype=np.float64)
    for src in inactive_idx:
        out[int(src), col_pos[int(src)]] = 1.0
    if n_keep_active > 0:
        out[
            np.ix_(
                active_idx,
                np.asarray([col_pos[int(j)] for j in keep_active_idx], dtype=int),
            )
        ] = T_active
    return out, rank


def apply_coefficient_transform(B, penalties, T):
    B = np.asarray(B, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    B_new = B @ T
    out_penalties = []
    for S in penalties:
        S = np.asarray(S, dtype=np.float64)
        S_new = 0.5 * (T.T @ S @ T + (T.T @ S @ T).T)
        out_penalties.append(S_new)
    return B_new, out_penalties


__all__ = [
    "dependent_column_indices",
    "orthogonal_residual",
    "independent_column_indices",
    "null_space_basis_from_constraint_matrix",
    "localized_null_space_basis_from_constraint_matrix",
    "apply_coefficient_transform",
]
