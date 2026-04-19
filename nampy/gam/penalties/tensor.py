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


def tensor_penalty_rescale_factors(B, penalties, tol=1e-12, *, x_norm_axis="row"):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    if len(penalties) == 0:
        return []
    if x_norm_axis == "row":
        x_scale = float(np.max(np.sum(np.abs(B), axis=1)) ** 2)
    elif x_norm_axis == "col":
        x_scale = float(np.max(np.sum(np.abs(B), axis=0)) ** 2)
    else:
        raise ValueError("x_norm_axis must be 'row' or 'col'.")
    if x_scale <= tol:
        return [1.0] * len(penalties)
    out = []
    for S in penalties:
        # Mirror mgcv/R/smooth.r::smoothCon(), which rescales each penalty by
        # `norm(sm$S[[i]]) / norm(sm$X, type="I")^2`. For matrices `norm()`
        # defaults to the one-norm (maximum column sum), not the infinity norm.
        s_scale = float(np.linalg.norm(S, ord=1)) / x_scale
        out.append(1.0 if s_scale <= tol else s_scale)
    return out


def rescale_tensor_penalties_for_fit(
    B,
    penalties,
    tol=1e-12,
    *,
    x_norm_axis="row",
    return_scales=False,
):
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    scales = tensor_penalty_rescale_factors(
        B, penalties, tol=tol, x_norm_axis=x_norm_axis
    )
    out = [
        S.copy() if float(scale) <= tol else S / float(scale)
        for S, scale in zip(penalties, scales)
    ]
    if return_scales:
        return out, scales
    return out


__all__ = [
    "lifted_tensor_penalty",
    "tensor_product_penalties",
    "normalize_tensor_marginal_penalty",
    "tensor_penalty_rescale_factors",
    "rescale_tensor_penalties_for_fit",
]
