import numpy as np
from scipy.linalg import eigh

from .._mgcv_constants import EIG_TOL_POWER
from ..penalties.algebra import penalty_eigendecomposition


def rowwise_kronecker(matrices):
    mats = [np.asarray(M, dtype=np.float64) for M in matrices]
    if len(mats) == 0:
        raise ValueError("matrices must contain at least one matrix.")
    n = mats[0].shape[0]
    for M in mats:
        if M.ndim != 2 or M.shape[0] != n:
            raise ValueError("All marginal model matrices must be 2D with equal rows.")
    out = mats[0]
    for M in mats[1:]:
        out = np.einsum("ij,ik->ijk", out, M, optimize=True).reshape(
            n, out.shape[1] * M.shape[1]
        )
    return out


def _eigen_split(raw_basis, raw_penalty, tol=1e-10, *, mode="range_null"):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)

    if mode == "range_null":
        dec = penalty_eigendecomposition(S, tol=tol)
        U0, U1, d_pos = dec["U0"], dec["U1"], dec["d_pos"]
        if d_pos.size > 0:
            T_r = U1 / np.sqrt(d_pos)[np.newaxis, :]
            B_r = X @ T_r
        else:
            T_r = np.empty((S.shape[0], 0), dtype=np.float64)
            B_r = np.empty((X.shape[0], 0), dtype=np.float64)
        T_n = U0
        B_n = (
            X @ T_n if T_n.shape[1] > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        return {
            "B_range": B_r,
            "B_null": B_n,
            "T_range": T_r,
            "T_null": T_n,
            "range_dim": B_r.shape[1],
            "null_dim": B_n.shape[1],
            "rank": dec["rank"],
            "null_space_dim": dec["null_space_dim"],
            "tol_eff": dec["tol_eff"],
        }

    if mode != "t2":
        raise ValueError(f"Unknown eigen split mode {mode!r}.")

    p = X.shape[1]
    evals, U = eigh(0.5 * (S + S.T), driver="evr")
    idx = np.argsort(evals)[::-1]
    evals, U = evals[idx], U[:, idx]

    tol_eff = float(np.finfo(np.float64).eps) ** EIG_TOL_POWER * max(
        1.0, float(np.max(evals)) if evals.size else 1.0
    )
    rank = int(np.sum(evals > tol_eff))
    null_exists = rank < p

    E = np.ones(p, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    Xp = X @ U
    col_norm = np.sum(Xp**2, axis=0) / (E**2)
    av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0

    if null_exists:
        for i in range(rank, p):
            if av_norm > 0.0 and col_norm[i] > 0.0:
                E[i] = np.sqrt(col_norm[i] / av_norm)

    P = U / E[np.newaxis, :]
    Xp = Xp / E[np.newaxis, :]

    if null_exists and rank < p - 1:
        ind = list(range(rank, p))
        rind = list(range(p - 1, rank - 1, -1))
        Xn = Xp[:, ind].copy()
        n = Xn.shape[0]
        one = np.ones(n, dtype=np.float64)
        Xn -= (one[:, None] * (one[None, :] @ Xn)) / n
        um_evals, um_vecs = eigh(Xn.T @ Xn, driver="evr")
        desc = np.argsort(um_evals)[::-1]
        um_vecs = um_vecs[:, desc]
        Xp[:, rind] = Xp[:, ind] @ um_vecs
        P[:, rind] = P[:, ind] @ um_vecs

    if rank > 0:
        pen_idx = list(range(rank))
        scale = 1.0 / np.sqrt(float(np.mean(Xp[:, pen_idx] ** 2)))
        Xp[:, pen_idx] *= scale
        P[pen_idx, :] *= scale

    if null_exists:
        null_idx = list(range(rank, p))
        scale_f = 1.0 / np.sqrt(float(np.mean(Xp[:, null_idx] ** 2)))
        Xp[:, null_idx] *= scale_f
        P[null_idx, :] *= scale_f

    B_r = Xp[:, :rank] if rank > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
    B_n = Xp[:, rank:] if null_exists else np.empty((X.shape[0], 0), dtype=np.float64)
    T_r = P[:, :rank] if rank > 0 else np.empty((p, 0), dtype=np.float64)
    T_n = P[:, rank:] if null_exists else np.empty((p, 0), dtype=np.float64)

    return {
        "B_range": B_r,
        "B_null": B_n,
        "T_range": T_r,
        "T_null": T_n,
        "range_dim": int(B_r.shape[1]),
        "null_dim": int(B_n.shape[1]),
        "rank": rank,
        "null_space_dim": int(p - rank),
        "tol_eff": tol_eff,
    }


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="range_null")


def t2_marginal_reparameterization(raw_basis, raw_penalty, tol=1e-10, *, knots=None):
    del knots
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="t2")


__all__ = [
    "rowwise_kronecker",
    "marginal_range_null_decomposition",
    "t2_marginal_reparameterization",
]
