"""Reparameterized penalty / Laplace helpers for ML/REML criteria."""

import numpy as np

from ..._model_state import (
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ..reparam import (
    build_estimate_gam_setup_state,
    ensure_penalty_reparameterization_state,
    sl_group_indices,
    sl_lambda_vector,
)


def _require_sl_reparam_state(model):
    state = ensure_penalty_reparameterization_state(model)
    if not hasattr(state, "sl_blocks"):
        raise RuntimeError(
            "Laplace mixed-model Sl state is unavailable; exact UrS "
            "reparameterization state cannot be used for Sl helpers."
        )
    return state


def _laplace_lambda_vector(model, sp):
    state = _require_sl_reparam_state(model)
    return sl_lambda_vector(state, sp)


def _lambda_group_indices(model):
    state = _require_sl_reparam_state(model)
    groups = sl_group_indices(state)
    if groups is None:
        return {}
    return {
        int(sp_idx): np.asarray(idxs, dtype=np.int64) for sp_idx, idxs in groups.items()
    }


def _penalty_derivative_matrices(model, sp):
    setup = build_estimate_gam_setup_state(model)
    n_full = int(np.asarray(setup.X, dtype=np.float64).shape[1])
    mats = [
        np.zeros((n_full, n_full), dtype=np.float64)
        for _ in range(int(_n_smoothing_params(model) or 0))
    ]
    if not mats:
        return mats

    for pb, S_local, off_i in zip(
        _penalty_blocks_seq(model),
        list(setup.S),
        np.asarray(setup.off, dtype=np.int64),
        strict=True,
    ):
        k = int(pb.smoothing_index)
        S_local = np.asarray(S_local, dtype=np.float64)
        start = int(off_i) - 1
        stop = start + int(S_local.shape[0])
        mats[k][start:stop, start:stop] += float(sp[k]) * S_local
    return mats
