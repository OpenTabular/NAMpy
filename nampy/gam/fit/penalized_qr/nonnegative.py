"""
Nonnegative-weight penalized QR state construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from ..linalg.stacked_qr import (
    _dorgqr_economic,
    _dormqr_apply,
    _stacked_penalized_ls_nonneg_solution,
)
from ..linalg.matrix_reindexing import (
    drop_columns_dense,
    drop_rows_dense,
    permute_columns,
    permute_rows,
)


@dataclass(frozen=True)
class NonnegativePenalizedQRState:
    rank: int
    n_drop: int
    drop: np.ndarray
    pivot1: np.ndarray
    kept_original_indices: tuple[int, ...]
    P: np.ndarray
    K: np.ndarray
    Rh: np.ndarray
    PKtz: np.ndarray
    ldet_XWX_plus_S: float
    beta_full: np.ndarray
    rS_work: np.ndarray


def _validate_inputs(X, z, w, penalty_sqrt_E, penalty_rank_Es, rS):
    if np.any(np.asarray(w, dtype=np.float64) < 0):
        raise ValueError("build_penalized_qr_state_nonnegative requires non-negative weights.")
    X = np.asarray(X, dtype=np.float64, order="C")
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n, q = X.shape
    if z.shape[0] != n or w.shape[0] != n:
        raise ValueError("z and w must have length n (rows of X).")
    E = np.asarray(penalty_sqrt_E, dtype=np.float64, order="C")
    Es = np.asarray(penalty_rank_Es, dtype=np.float64, order="C")
    rS = np.asarray(rS, dtype=np.float64, order="C")
    if E.ndim != 2 or Es.ndim != 2:
        raise ValueError("penalty_sqrt_E and penalty_rank_Es must be 2-D.")
    enrow = int(E.shape[0])
    if E.shape[1] != q or Es.shape[1] != q or Es.shape[0] != enrow:
        raise ValueError("E and Es must have q columns matching X.")
    if rS.shape[0] != q:
        raise ValueError("rS must have q rows.")
    return X, z, w, E, Es, rS


def _build_q1_and_k(out, n, rr, rank):
    qr_aug_f = np.asfortranarray(np.asarray(out.qr_aug, dtype=np.float64).copy())
    Q_aug = _dorgqr_economic(qr_aug_f, out.tau_aug, rank)
    Q_top = np.asarray(Q_aug[:rr, :], dtype=np.float64)
    Q1 = np.zeros((n, rank), dtype=np.float64, order="F")
    Q1[:rr, :] = Q_top
    Q1 = _dormqr_apply(b"L", b"N", out.qr_wx, out.tau_wx, Q1)
    return np.asarray(Q1, dtype=np.float64, order="C"), qr_aug_f


def _compute_pk_rhs(Rh, R_nr_rank, K, X, z, w, drop, pivot1):
    P = solve_triangular(Rh, np.eye(Rh.shape[0], dtype=np.float64), lower=False)
    raw = np.sqrt(np.maximum(w, 0.0))
    zz = z * raw
    ktz = K.T @ zz
    q1tz = K.T @ zz
    wz = w * z
    X_drop = drop_columns_dense(X, drop)
    X_piv = permute_columns(X_drop, pivot1, reverse=False)
    xwz = X_piv.T @ wz

    norm1 = 0.0
    norm2 = 0.0
    rank = Rh.shape[0]
    for i in range(rank):
        s = 0.0
        for j in range(i + 1):
            s += R_nr_rank[j, i] * q1tz[j]
        diff = s - xwz[i]
        norm1 += diff * diff
        norm2 += xwz[i] * xwz[i]
    return P, ktz, xwz, norm1, norm2


def _scatter_coefficients(PKtz, kept, pivot1, q):
    coef_nat = np.zeros(PKtz.shape[0], dtype=np.float64)
    coef_nat[pivot1] = PKtz
    beta_full = np.zeros(q, dtype=np.float64)
    for j, orig_col in enumerate(kept):
        beta_full[orig_col] = coef_nat[j]
    return beta_full


def build_penalized_qr_state_nonnegative(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    penalty_sqrt_E: np.ndarray,
    penalty_rank_Es: np.ndarray,
    rS: np.ndarray,
    rank_tol: float,
    reml: bool,
    Mp: int = 0,
) -> NonnegativePenalizedQRState:
    X, z, w, E, Es, rS = _validate_inputs(X, z, w, penalty_sqrt_E, penalty_rank_Es, rS)
    n, q = X.shape

    out = _stacked_penalized_ls_nonneg_solution(
        X,
        z,
        w,
        penalty_sqrt=E,
        penalty_rank_rows=Es,
        P_dense=None,
        rank_tol=rank_tol,
        coef_method="householder",
        near_singular_null_pin=False,
    )

    rank = int(out.system_rank)
    rr = int(out.n_wx_econ)
    drop = np.asarray(out.dropped_column_indices, dtype=int)
    n_drop = int(drop.size)
    pivot1 = np.asarray(out.pivot_aug, dtype=int).ravel()
    if pivot1.size != rank:
        raise RuntimeError("internal: pivot_aug length must equal system rank.")

    kept = tuple(int(k) for k in out.kept_original_indices)
    n_pen = int(E.shape[0])
    nr = rr + n_pen

    K, qr_aug_f = _build_q1_and_k(out, n, rr, rank)
    R_nr_rank = np.triu(np.asarray(qr_aug_f[:nr, :rank], dtype=np.float64))
    Rh = np.triu(np.asarray(out.upper_r_final, dtype=np.float64))

    P, ktz, xwz, norm1, norm2 = _compute_pk_rhs(
        Rh, R_nr_rank, K, X, z, w, drop, pivot1
    )
    if norm1 > rank_tol * norm2:
        tmp = solve_triangular(Rh, xwz, lower=False, trans="T")
        PKtz = solve_triangular(Rh, tmp, lower=False)
    else:
        PKtz = solve_triangular(Rh, ktz, lower=False)

    if reml:
        d = np.abs(np.diag(Rh))
        ldet = float(2.0 * np.sum(np.log(np.maximum(d, np.finfo(np.float64).tiny))))
    else:
        ldet = 0.0

    beta_full = _scatter_coefficients(PKtz, kept, pivot1, q)
    rS_drop = drop_rows_dense(rS, drop) if n_drop else rS.copy()
    rS_work = permute_rows(rS_drop, pivot1, reverse=False)

    _ = Mp
    return NonnegativePenalizedQRState(
        rank=rank,
        n_drop=n_drop,
        drop=drop.copy(),
        pivot1=pivot1.copy(),
        kept_original_indices=kept,
        P=P,
        K=K,
        Rh=Rh,
        PKtz=np.asarray(PKtz, dtype=np.float64).ravel().copy(),
        ldet_XWX_plus_S=ldet,
        beta_full=beta_full,
        rS_work=rS_work,
    )
