from __future__ import annotations

import numpy as np
from scipy.linalg import qr as scipy_qr


def orthogonal_residual(B, A):
    B = np.asarray(B, dtype=np.float64)
    if A is None or np.asarray(A).size == 0 or np.asarray(A).shape[1] == 0:
        return B
    A = np.asarray(A, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    return B - A @ coef


def independent_column_indices(B, A=None, tol: float = 1e-10):
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")
    if B.shape[1] == 0:
        return np.array([], dtype=int)
    Rb = orthogonal_residual(B, A)
    if np.all(np.abs(Rb) <= tol):
        return np.array([], dtype=int)
    _Q, R, piv = scipy_qr(Rb, mode="economic", pivoting=True)
    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return np.array([], dtype=int)
    rank_tol = max(B.shape) * np.finfo(float).eps * diag_R[0]
    tol_eff = max(float(tol), float(rank_tol))
    rank = int(np.sum(diag_R > tol_eff))
    if rank <= 0:
        return np.array([], dtype=int)
    return np.sort(np.asarray(piv[:rank], dtype=int))


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
    "orthogonal_residual",
    "independent_column_indices",
    "null_space_basis_from_constraint_matrix",
    "localized_null_space_basis_from_constraint_matrix",
    "apply_coefficient_transform",
]
