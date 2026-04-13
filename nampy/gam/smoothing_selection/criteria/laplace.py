"""Reparameterized penalty / Laplace helpers for ML/REML criteria."""

import numpy as np

from ..._model_state import _coef_column_offset
from ..reparam import (
    ensure_penalty_reparameterization_state,
    sl_group_indices,
    sl_lambda_vector,
)


def _laplace_lambda_vector(model, sp):
    state = ensure_penalty_reparameterization_state(model)
    return sl_lambda_vector(state, sp)


def _lambda_group_indices(model):
    state = ensure_penalty_reparameterization_state(model)
    groups = sl_group_indices(state)
    if groups is None:
        return {}
    return {
        int(sp_idx): np.asarray(idxs, dtype=np.int64) for sp_idx, idxs in groups.items()
    }


def _penalty_derivative_matrices(model, sp):
    off = _coef_column_offset(model)
    n_full = int(model.n_coef_ + off)
    offset0 = off
    mats = [
        np.zeros((n_full, n_full), dtype=np.float64)
        for _ in range(int(model.n_smoothing_params_ or 0))
    ]
    if not mats:
        return mats

    for pb in model.penalty_blocks_:
        k = int(pb.smoothing_index)
        sl = pb.coef_slice
        full_sl = slice(offset0 + sl.start, offset0 + sl.stop)
        mats[k][full_sl, full_sl] += float(sp[k]) * np.asarray(
            pb.matrix, dtype=np.float64
        )
    return mats
