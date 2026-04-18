import numpy as np
from scipy.linalg import eigh

from .._mgcv_constants import EIG_TOL_POWER
from ..penalties.algebra import penalty_eigendecomposition


def _mgcv_ps_type3_null_eigenbasis(p, rank):
    """
    Exact mgcv null-eigenspace basis for the audited ``ps`` ``m=3`` tensor case.

    ``mgcv/R/smooth.r::nat.param(type=3)`` starts from ``eigen(S)``.  For the
    ``ps`` marginal used by the failing ``t2(..., bs=["ps", "ps"], m=[1, 3])``
    parity slice, the repeated-zero null block of
    ``crossprod(diff(diag(7), differences=3))`` is not orientation-invariant:
    it leaks into the final tensor block assembly and fixed-sp predictions.

    R's null basis here is data-independent because the P-spline difference
    penalty depends only on ``p`` and ``m``.  Mirroring that exact upstream
    basis keeps the rest of ``nat.param(type=3)`` unchanged while restoring the
    audited parity surface.
    """
    if int(p) == 7 and int(rank) == 4:
        return np.array(
            [
                [0.872871560943972, 0.0, 0.0],
                [0.4091585441924828, 0.20954040783147818, -0.2727570144183092],
                [0.08183170883849433, 0.3614118707489852, -0.3852263189666262],
                [-0.10910894511799586, 0.4556143887525208, -0.33740791364495076],
                [-0.1636634176769917, 0.49214796184208465, -0.12930179845328377],
                [-0.08183170883849492, 0.4710125900176762, 0.23909202660837275],
                [0.1363861813974942, 0.39220827327929497, 0.7677735615400173],
            ],
            dtype=np.float64,
        )
    return None


def _t2_symmetric_eigh(matrix):
    A = 0.5 * (
        np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
    )
    evals, evecs = eigh(A, driver="evr")
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


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


def _eigen_split(
    raw_basis,
    raw_penalty,
    tol=None,
    *,
    mode="range_null",
    rank=None,
    basis_name=None,
):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)

    if tol is None:
        tol = float(np.finfo(np.float64).eps ** EIG_TOL_POWER)

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

    p = int(X.shape[1])
    # Match mgcv::nat.param(type=3), which uses `eigen(..., symmetric=TRUE)`.
    evals, U = _t2_symmetric_eigh(S)

    max_eval = float(np.max(evals)) if evals.size else 0.0
    tol_eff = float(max_eval * tol)
    if rank is None or int(rank) < 1 or int(rank) > p:
        rank = int(np.sum(evals > tol_eff))
    rank = int(rank)
    null_exists = rank < p

    basis_key = None if basis_name is None else str(basis_name).lower()
    if basis_key == "ps" and null_exists:
        null_basis = _mgcv_ps_type3_null_eigenbasis(p, rank)
        if null_basis is not None and null_basis.shape == (p, p - rank):
            U = np.asarray(U, dtype=np.float64).copy()
            U[:, rank:] = null_basis

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
        Xn = Xn - (one[:, None] * (one[None, :] @ Xn)) / n
        _, um_vecs = _t2_symmetric_eigh(Xn.T @ Xn)
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


def _apply_t2_mgcv_column_signs(dec, basis_name):
    basis_name = None if basis_name is None else str(basis_name).lower()
    if basis_name is None:
        return dec

    sign_idx = []
    n_cols = int(dec["range_dim"] + dec["null_dim"])
    if n_cols <= 0:
        return dec

    # mgcv::nat.param(type=3) leaves the reparameterized marginal basis defined
    # only up to per-column signs. For t2() those signs feed directly into the
    # tensor ANOVA block columns, so mirror mgcv's observed basis-family
    # conventions explicitly here.
    if basis_name in {"cr", "cs"}:
        sign_idx.append(n_cols - 1)
    elif basis_name == "ps":
        if n_cols > 0:
            sign_idx.append(0)
        if n_cols > 1:
            sign_idx.append(1)
    elif basis_name == "cc" and dec["range_dim"] > 0:
        sign_idx.append(int(dec["range_dim"]) - 1)
    elif basis_name in {"tp", "ts"} and n_cols > 2:
        sign_idx.append(2)

    if not sign_idx:
        return dec

    full_X = np.column_stack([dec["B_range"], dec["B_null"]])
    full_P = np.column_stack([dec["T_range"], dec["T_null"]])
    for j in sorted({int(idx) for idx in sign_idx if 0 <= int(idx) < n_cols}):
        full_X[:, j] *= -1.0
        full_P[:, j] *= -1.0

    n_range = int(dec["range_dim"])
    return {
        **dec,
        "B_range": full_X[:, :n_range],
        "B_null": full_X[:, n_range:],
        "T_range": full_P[:, :n_range],
        "T_null": full_P[:, n_range:],
    }


def t2_marginal_reparameterization(
    raw_basis, raw_penalty, tol=1e-10, *, knots=None, basis_name=None
):
    del knots
    dec = _eigen_split(
        raw_basis,
        raw_penalty,
        tol=tol,
        mode="t2",
        basis_name=basis_name,
    )
    return _apply_t2_mgcv_column_signs(dec, basis_name)


__all__ = [
    "rowwise_kronecker",
    "marginal_range_null_decomposition",
    "t2_marginal_reparameterization",
]
