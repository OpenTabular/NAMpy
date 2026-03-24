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


def null_space_basis_from_constraint_matrix(C, d: int | None = None, tol: float = 1e-10):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError("Constraint matrix must be 2D.")
    if d is not None and C.shape[1] != int(d):
        raise ValueError(f"Constraint matrix has width {C.shape[1]}, expected {d}.")
    if C.size == 0:
        return np.eye(C.shape[1], dtype=np.float64), 0
    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    if s.size == 0:
        return np.eye(C.shape[1], dtype=np.float64), 0
    tol_eff = max(float(tol), float(np.max(s)) * max(C.shape) * np.finfo(float).eps)
    rank = int(np.sum(s > tol_eff))
    return Vt[rank:, :].T.copy(), rank


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
    "apply_coefficient_transform",
]
