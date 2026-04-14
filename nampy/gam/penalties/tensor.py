"""Tensor-product penalty helpers owned by the penalty layer."""

from __future__ import annotations

import numpy as np


def lifted_tensor_penalty(S, basis_dims, axis):
    S = np.asarray(S, dtype=np.float64)
    basis_dims = [int(d) for d in basis_dims]
    left_dim = int(np.prod(basis_dims[:axis], dtype=np.int64)) if axis > 0 else 1
    right_dim = (
        int(np.prod(basis_dims[axis + 1 :], dtype=np.int64))
        if axis + 1 < len(basis_dims)
        else 1
    )
    out = S
    if left_dim > 1:
        out = np.kron(np.eye(left_dim, dtype=np.float64), out)
    if right_dim > 1:
        out = np.kron(out, np.eye(right_dim, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def tensor_product_penalties(marginal_penalties, basis_dims):
    return [
        lifted_tensor_penalty(S, basis_dims=basis_dims, axis=j)
        for j, S in enumerate(marginal_penalties)
    ]


def normalize_tensor_marginal_penalty(S, tol=1e-12):
    S = np.asarray(S, dtype=np.float64)
    if S.shape[0] == 0:
        return S.copy()
    evals = np.linalg.eigvalsh(0.5 * (S + S.T))
    scale = float(np.max(evals))
    if scale <= tol:
        return S.copy()
    return S / scale


def rescale_tensor_penalties_for_fit(B, penalties, tol=1e-12):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    if len(penalties) == 0:
        return []
    x_scale = float(np.max(np.sum(np.abs(B), axis=1)) ** 2)
    if x_scale <= tol:
        return [S.copy() for S in penalties]
    out = []
    for S in penalties:
        s_scale = float(np.max(np.sum(np.abs(S), axis=0))) / x_scale
        out.append(S.copy() if s_scale <= tol else S / s_scale)
    return out


__all__ = [
    "lifted_tensor_penalty",
    "tensor_product_penalties",
    "normalize_tensor_marginal_penalty",
    "rescale_tensor_penalties_for_fit",
]
