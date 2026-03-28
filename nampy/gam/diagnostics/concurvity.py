from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def _term_blocks_for_concurvity(model, X):
    offset = 1 if bool(getattr(model, "fit_intercept", False)) else 0
    q = int(X.shape[1])
    covered = np.zeros(q, dtype=bool)
    blocks = []

    for tb in getattr(model, "term_blocks_", ()) or ():
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        idx = np.arange(
            int(tb.coef_slice.start) + offset,
            int(tb.coef_slice.stop) + offset,
            dtype=int,
        )
        if idx.size == 0:
            continue
        covered[idx] = True
        blocks.append((str(tb.label), idx))

    para_idx = np.flatnonzero(~covered)
    if para_idx.size > 0:
        blocks.insert(0, ("para", para_idx))

    if len(blocks) == 0:
        raise ValueError("No smooth or parametric components available for concurvity.")
    return blocks


def _full_coef_vector(model) -> np.ndarray:
    beta = np.asarray(model.coef_, dtype=np.float64).ravel()
    if bool(getattr(model, "fit_intercept", False)):
        return np.concatenate(
            [np.array([float(model.intercept_)], dtype=np.float64), beta]
        )
    return beta


def _qr_R(X: np.ndarray) -> np.ndarray:
    return np.linalg.qr(np.asarray(X, dtype=np.float64), mode="reduced")[1]


def concurvity(model, full: bool = True):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    Z = np.asarray(model.design_.design_matrix, dtype=np.float64)
    if bool(getattr(model, "fit_intercept", False)):
        X = np.column_stack([np.ones(Z.shape[0], dtype=np.float64), Z])
    else:
        X = Z
    X = X[np.sum(np.isnan(X), axis=1) == 0, :]
    X = _qr_R(X)
    blocks = _term_blocks_for_concurvity(model, X)
    beta_full = _full_coef_vector(model)

    labels = [lab for lab, _ in blocks]
    n_terms = len(blocks)
    measure_names = ("worst", "observed", "estimate")

    if full:
        out = np.zeros((3, n_terms), dtype=np.float64)
        for i, (_lab_i, idx_i) in enumerate(blocks):
            mask = np.ones(int(X.shape[1]), dtype=bool)
            mask[idx_i] = False
            Xi = X[:, mask]
            Xj = X[:, idx_i]
            r = int(Xi.shape[1])
            R = _qr_R(np.column_stack([Xi, Xj]))[:, r:]
            Rt = _qr_R(R)
            F = solve_triangular(Rt.T, R[:r, :].T, lower=True)
            out[0, i] = float(np.linalg.svd(F, compute_uv=False)[0] ** 2)
            beta = beta_full[idx_i]
            denom = float(np.sum((Rt @ beta) ** 2))
            out[1, i] = 0.0 if denom <= 0.0 else float(np.sum((R[:r, :] @ beta) ** 2) / denom)
            out[2, i] = float(np.sum(R[:r, :] ** 2) / np.sum(R**2))
        return {
            "measure_names": measure_names,
            "labels": labels,
            "values": out,
        }

    mats = [np.ones((n_terms, n_terms), dtype=np.float64) for _ in range(3)]
    for i, (_lab_i, idx_i) in enumerate(blocks):
        Xi = X[:, idx_i]
        r = int(Xi.shape[1])
        for j, (_lab_j, idx_j) in enumerate(blocks):
            if i == j:
                continue
            Xj = X[:, idx_j]
            R = _qr_R(np.column_stack([Xi, Xj]))[:, r:]
            Rt = _qr_R(R)
            F = solve_triangular(Rt.T, R[:r, :].T, lower=True)
            mats[0][i, j] = float(np.linalg.svd(F, compute_uv=False)[0] ** 2)
            beta = beta_full[idx_j]
            denom = float(np.sum((Rt @ beta) ** 2))
            mats[1][i, j] = 0.0 if denom <= 0.0 else float(np.sum((R[:r, :] @ beta) ** 2) / denom)
            mats[2][i, j] = float(np.sum(R[:r, :] ** 2) / np.sum(R**2))

    return {
        "measure_names": measure_names,
        "labels": labels,
        "values": {
            name: mat for name, mat in zip(measure_names, mats)
        },
    }


__all__ = ["concurvity"]
