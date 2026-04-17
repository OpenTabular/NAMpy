"""Penalty assembly helpers for fitted GAM models."""

from __future__ import annotations

import numpy as np

from .._model_state import (
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
    _term_blocks_seq,
)


def one_penalty_per_term_matrices(model):
    penalties = []
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    for tb in _term_blocks_seq(model):
        matches = [pb for pb in penalty_blocks if pb.coef_slice == tb.coef_slice]
        if not matches:
            penalties.append(
                np.zeros((tb.basis_train.shape[1], tb.basis_train.shape[1]))
            )
            continue
        P_term = np.zeros_like(np.asarray(matches[0].matrix, dtype=np.float64))
        for pb in matches:
            P_term += np.asarray(pb.matrix, dtype=np.float64)
        penalties.append(P_term)
    return penalties


def assemble_penalty_matrix(model, smoothing_params):
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64)
    n_coef = _n_coef(model)
    n_smoothing_params = _n_smoothing_params(model)
    if smoothing_params.shape != (n_smoothing_params,):
        raise ValueError(
            f"Expected {n_smoothing_params} smoothing parameters, "
            f"got shape {smoothing_params.shape}."
        )
    P = np.zeros((n_coef, n_coef), dtype=np.float64)
    for pb in _penalty_blocks_seq(model):
        sl = pb.coef_slice
        lam = float(smoothing_params[pb.smoothing_index])
        P[sl, sl] += lam * pb.matrix
    return P
