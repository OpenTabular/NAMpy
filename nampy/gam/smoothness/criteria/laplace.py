"""Reparameterized penalty / Laplace helpers for ML/REML criteria."""
import numpy as np

def _ensure_penalty_reparameterization(model):
    if (
        model.X_fix_ is None
        or model.Z_rand_ is None
        or getattr(model, "_reparam_sp_groups_", None) is None
    ):
        model._build_penalty_reparameterized_system()


def _laplace_lambda_vector(model, sp):
    blocks = getattr(model, "_reparam_rand_blocks_", None)
    if not blocks:
        return np.empty((0,), dtype=np.float64)
    lam_parts = []
    for block in blocks:
        n_pen = int(block["n_pen"])
        if n_pen == 0:
            continue
        sp_val = float(sp[int(block["smoothing_index"])])
        scaling = float(block.get("lambda_scaling", 1.0))
        lam_val = sp_val * scaling
        lam_parts.append(np.full(n_pen, lam_val, dtype=np.float64))
    return np.concatenate(lam_parts) if lam_parts else np.empty((0,), dtype=np.float64)


def _lambda_group_indices(model):
    groups = getattr(model, "_reparam_sp_groups_", None)
    if groups is None:
        return {}
    return {
        int(sp_idx): np.asarray(idxs, dtype=np.int64)
        for sp_idx, idxs in groups.items()
    }


def _penalty_derivative_matrices(model, sp):
    n_full = int(model.n_coef_ + (1 if model.fit_intercept else 0))
    offset0 = 1 if model.fit_intercept else 0
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
        mats[k][full_sl, full_sl] += float(sp[k]) * np.asarray(pb.matrix, dtype=np.float64)
    return mats

