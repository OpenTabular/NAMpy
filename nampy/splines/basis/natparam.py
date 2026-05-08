"""Natural-parameterization helpers shared by smooth constructors."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from ...gam.linalg import matrix_is_rank_deficient, symmetric_eigh


def nat_param_type0(X, S, rank=None, tol=None, unit_fnorm=True):
    """
    Python implementation of ``mgcv::nat.param(X, S, type=0)``.

    Returns a dict with transformed model matrix ``X``, positive diagonal
    penalty entries ``D``, coefficient back-transform ``P``, and penalty rank.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tol = np.finfo(float).eps**0.8 if tol is None else float(tol)

    Q, R = scipy_qr(X, mode="economic", pivoting=False)
    if matrix_is_rank_deficient(R):
        raise ValueError(
            "Model matrix is not full rank in natural-parameter construction."
        )

    eye = np.eye(R.shape[0], dtype=np.float64)
    invR = solve_triangular(R, eye, lower=False, check_finite=False)
    RSR = invR.T @ S @ invR
    RSR = 0.5 * (RSR + RSR.T)

    evals, U = symmetric_eigh(RSR, descending=True, use_scipy=True)

    if rank is None or rank < 1 or rank > S.shape[0]:
        rank = int(np.sum(evals > np.max(evals) * tol))

    D = evals[:rank].copy()
    Xn = Q @ U
    P = invR @ U

    if unit_fnorm:
        if rank > 0:
            ind = np.arange(rank)
            scale = 1.0 / np.sqrt(np.mean(Xn[:, ind] ** 2))
            Xn[:, ind] *= scale
            P[:, ind] *= scale
            D *= scale**2

        if rank < Xn.shape[1]:
            ind = np.arange(rank, Xn.shape[1])
            scalef = 1.0 / np.sqrt(np.mean(Xn[:, ind] ** 2))
            Xn[:, ind] *= scalef
            P[:, ind] *= scalef

    return {
        "X": Xn,
        "D": D,
        "P": P,
        "rank": int(rank),
    }


def nat_param_type1(X, S, rank=None, tol=None, unit_fnorm=True):
    """
    Python implementation of ``mgcv::nat.param(X, S, type=1)``.

    This reparameterizes so that the penalty in the penalized columns is the
    identity. Returns the same dictionary structure as :func:`nat_param_type0`.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tol = np.finfo(float).eps**0.8 if tol is None else float(tol)

    # Mirror mgcv/R/smooth.r::nat.param(type=1): base R QR followed by
    # eigen(RSR, symmetric=TRUE). The tiny triangle asymmetry in RSR fixes the
    # orientation of degenerate null-space columns for factor-smooth penalties,
    # so do not explicitly symmetrize it here.
    Q, R = scipy_qr(X, mode="economic", pivoting=False, check_finite=False)
    if matrix_is_rank_deficient(R):
        raise ValueError(
            "Model matrix is not full rank in natural-parameter construction."
        )

    tmp = solve_triangular(R.T, S.T, lower=True, check_finite=False)
    RSR = solve_triangular(R.T, tmp.T, lower=True, check_finite=False)
    evals_asc, U_asc = scipy_eigh(
        RSR,
        driver="ev",
        lower=True,
        check_finite=False,
    )

    if rank is None or rank < 1 or rank > S.shape[0]:
        max_eval = np.max(evals_asc) if evals_asc.size else 0.0
        thresh = max_eval * tol
        rank = int(np.sum(evals_asc > thresh))
    rank = max(0, min(rank, S.shape[0]))

    order_asc = np.argsort(evals_asc)
    order = np.concatenate(
        [order_asc[::-1][:rank], order_asc[: max(0, evals_asc.size - rank)]]
    )
    evals = np.asarray(evals_asc[order], dtype=np.float64)
    U = np.asarray(U_asc[:, order], dtype=np.float64)

    D = evals[:rank].copy()
    Xn = Q @ U
    P = solve_triangular(R, U, lower=False, check_finite=False)

    total_cols = Xn.shape[1]
    E = np.ones(total_cols, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(D)
    Xn = Xn / E[np.newaxis, :]
    P = P / E[np.newaxis, :]
    D = np.ones(rank, dtype=np.float64)

    if unit_fnorm:
        if rank > 0:
            scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
            Xn[:, :rank] *= scale
            P[:, :rank] *= scale
            D *= scale**2

        if rank < Xn.shape[1]:
            scalef = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
            Xn[:, rank:] *= scalef
            P[:, rank:] *= scalef

    return {
        "X": Xn,
        "D": D,
        "P": P,
        "rank": int(rank),
    }


__all__ = ["nat_param_type0", "nat_param_type1"]
