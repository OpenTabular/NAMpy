from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from ..linalg.qr import r_linpack_qr_no_pivot, r_linpack_qr_r
from ..model_state import (
    _coef,
    _coef_column_offset,
    _coef_full,
    _compiled_model,
    _fit_intercept,
    _intercept,
    _require_fitted,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from ..predict.linear_predictor_matrix import build_lpmatrix
from ..term_labels import compiled_term_display_label, mgcv_term_display_label


def _term_indices_for_concurvity(model, n_coef: int):
    blocks = []
    smooth_starts = []

    for tb in _term_blocks_seq(model):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        if _compiled_model(model) is None:
            offset = _coef_column_offset(model)
            idx = np.arange(
                int(tb.coef_slice.start) + offset,
                int(tb.coef_slice.stop) + offset,
                dtype=int,
            )
        else:
            idx = _term_full_coefficient_indices(model, tb)
        idx = idx[(idx >= 0) & (idx < int(n_coef))]
        if idx.size == 0:
            continue
        smooth_starts.append(int(np.min(idx)))
        # mgcv's general-family matrices carry predictor-aware compact labels.
        # Ordinary NAMpy diagnostics have historically exposed the compiled
        # formula identity, including basis and dimension arguments.
        family_class = str(
            getattr(getattr(model, "family", None), "family_class", "")
        ).lower()
        label = (
            mgcv_term_display_label(tb)
            if family_class == "general"
            else compiled_term_display_label(tb)
        )
        blocks.append((label, idx))

    if len(blocks) == 0:
        raise ValueError("No smooth or parametric components available for concurvity.")

    # mgcv/R/mgcv.r::concurvity prepends a "para" block via:
    #   start <- c(1,start); stop <- c(min(start)-1,stop)
    # Because stop is computed after prepending 1, the upstream block is
    # effectively column 1 only, even when more parametric columns precede the
    # first smooth.
    first_smooth = min(smooth_starts)
    if first_smooth > 0:
        blocks.insert(0, ("para", np.array([0], dtype=int)))

    return blocks


def _full_coef_vector(model) -> np.ndarray:
    coef_full = _coef_full(model)
    if coef_full is not None:
        return np.asarray(coef_full, dtype=np.float64).ravel()
    beta = np.asarray(_coef(model), dtype=np.float64).ravel()
    if _fit_intercept(model):
        return np.concatenate(
            [np.array([float(_intercept(model))], dtype=np.float64), beta]
        )
    return beta


def _qr_R(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Concurvity QR input must be two-dimensional.")
    if X.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    # concurvity() explicitly requests base R's non-LAPACK, non-pivoted
    # ``qr(..., tol=0)`` at every stage.  A legal LAPACK QR has noticeably
    # different rounding for the nearly dependent multi-predictor spaces.
    packed, _qraux = r_linpack_qr_no_pivot(X)
    return r_linpack_qr_r(packed)


def _concurvity_measures(
    X_left: np.ndarray, X_right: np.ndarray, beta_right: np.ndarray
) -> tuple[float, float, float]:
    r = int(X_left.shape[1])
    if r == 0:
        return 0.0, 0.0, 0.0

    R = _qr_R(np.column_stack([X_left, X_right]))[:, r:]
    Rt = _qr_R(R)
    if Rt.size == 0:
        return 0.0, 0.0, 0.0

    leading = np.asarray(R[:r, :], dtype=np.float64)
    F = solve_triangular(Rt.T, leading.T, lower=True)
    worst = float(np.linalg.svd(F, compute_uv=False)[0] ** 2)

    beta_right = np.asarray(beta_right, dtype=np.float64).ravel()
    denom = float(np.sum((Rt @ beta_right) ** 2))
    observed = float(np.sum((leading @ beta_right) ** 2) / denom)

    total = float(np.sum(R**2))
    estimate = float(np.sum(leading**2) / total)
    return worst, observed, estimate


def concurvity(model, full: bool = True):
    _require_fitted(model)

    X = np.asarray(build_lpmatrix(model), dtype=np.float64)
    X = X[np.sum(np.isnan(X), axis=1) == 0, :]
    X = _qr_R(X)
    beta_full = _full_coef_vector(model)
    if beta_full.size != X.shape[1]:
        raise ValueError(
            "Concurvity requires coefficient and design columns to align exactly."
        )

    blocks = _term_indices_for_concurvity(model, X.shape[1])
    labels = [lab for lab, _idx in blocks]
    n_terms = len(labels)
    measure_names = ("worst", "observed", "estimate")

    if full:
        out = np.zeros((3, n_terms), dtype=np.float64)
        for i in range(n_terms):
            idx_i = np.asarray(blocks[i][1], dtype=int)
            keep = np.ones(X.shape[1], dtype=bool)
            keep[idx_i] = False
            Xi = X[:, keep]
            Xj = X[:, idx_i]
            beta = beta_full[idx_i]
            out[:, i] = _concurvity_measures(Xi, Xj, beta)
        return {
            "measure_names": measure_names,
            "labels": labels,
            "values": out,
        }

    mats = [np.ones((n_terms, n_terms), dtype=np.float64) for _ in range(3)]
    for i in range(n_terms):
        idx_i = np.asarray(blocks[i][1], dtype=int)
        Xi = X[:, idx_i]
        for j in range(n_terms):
            if i == j:
                continue
            idx_j = np.asarray(blocks[j][1], dtype=int)
            Xj = X[:, idx_j]
            beta = beta_full[idx_j]
            mats[0][i, j], mats[1][i, j], mats[2][i, j] = _concurvity_measures(
                Xi,
                Xj,
                beta,
            )

    return {
        "measure_names": measure_names,
        "labels": labels,
        "values": dict(zip(measure_names, mats, strict=True)),
    }


__all__ = ["concurvity"]
