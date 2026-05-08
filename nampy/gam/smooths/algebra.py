import numpy as np

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

    raise ValueError(f"Unknown eigen split mode {mode!r}.")


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="range_null")


__all__ = [
    "rowwise_kronecker",
    "marginal_range_null_decomposition",
]
