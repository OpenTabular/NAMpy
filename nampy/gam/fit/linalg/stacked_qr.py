"""
Stacked pivoted QR for Gaussian penalized least-squares.

When the penalized normal equations ``X'WX + S`` are singular or near-singular
(e.g. a random-effect term with a small smoothing parameter), a plain Cholesky
solve on the normal equations can pick an unstable coefficient vector.  This
module implements a numerically stable alternative using pivoted QR
decompositions through SciPy's supported interface, stacked as Wood (2017)
describes.

Algorithm outline
-----------------
1. QR-decompose ``sqrt(W) X`` with column pivoting to get an economy R factor.
2. Stack that R factor with the penalty square-root rows and QR-decompose again
   with column pivoting to reveal numerical rank and drop near-zero columns.
3. Solve for coefficients using the Householder/back-substitution chain from
   ``mgcv/src/gdi.c::pls_fit1``.
4. Reconstruct EDF-related matrices from the two QR factors.

Rank detection
--------------
Rank is revealed by an upper-triangular condition number estimate on the stacked
R factor.  The default threshold here is :data:`STACKED_QR_RANK_TOLERANCE`
(eps**0.66), but the `gam.fit3`-mirroring PIRLS and Gaussian exact paths pass
mgcv's ``rank.tol = .Machine$double.eps*100`` (mgcv/R/gam.fit3.r:131)
explicitly.
When ``penalty_blocks`` are provided, a Frobenius-normalised aggregate of the
unscaled penalty templates is used for rank detection (more numerically stable
than row-normalised ``sqrt(P)`` for mixed-scale penalties).

EDF computation
---------------
Effective degrees of freedom are computed as ``tr(F)`` where
``F = (R^{-1} K') (sqrt(W) X)`` and ``K`` comes from the Householder chain.
This matches the hat matrix trace produced by the coefficient post-processor
when using the same stacked QR path.

Fitting values ``X @ beta`` are numerically stable regardless of rank
deficiency. Raw coefficient vectors along ``null(X)`` may use a different
representative from R when the numerical backend selects a different pivot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.linalg import qr, solve_triangular

from ..._mgcv_constants import LOG_GUARD_MIN, QR_TOL_SCALE
from ...linalg import (
    balanced_penalty_template_sqrt_for_rank,
    symmetric_eigh,
    symmetrize_matrix,
    upper_triangular_condition_indicator,
)
from .matrix_reindexing import (
    drop_columns_dense,
    drop_rows_dense,
    permute_columns,
    permute_rows,
    restore_dropped_rows,
)

# Rank-detection threshold: condition number ratio above which a column is considered
# linearly dependent. Match mgcv's stacked-QR tolerance at eps**0.66.
STACKED_QR_RANK_TOLERANCE: float = float(
    np.finfo(np.float64).eps**0.66 * QR_TOL_SCALE
)


@dataclass(frozen=True)
class _StackedPlsNonnegOutcome:
    """Internal: one non-negative-weight stacked PLS solve + QR state for post-processing."""

    coef_full: np.ndarray
    eta: np.ndarray
    penalty_quadratic: float
    penalty_sqrt: np.ndarray
    weighted_X: np.ndarray
    system_rank: int
    upper_r_final: np.ndarray
    q_weighted: np.ndarray
    q_augmented: np.ndarray
    pivot_aug: np.ndarray
    n_wx_econ: int
    kept_original_indices: list[int]
    dropped_column_indices: np.ndarray
    covariance_rank_root: np.ndarray | None
    log_det_correction: float
    deviance_hessian_half: np.ndarray


@dataclass(frozen=True)
class NonnegativePenalizedQRState:
    rank: int
    n_drop: int
    drop: np.ndarray
    pivot1: np.ndarray
    kept_original_indices: tuple[int, ...]
    P: np.ndarray
    K: np.ndarray
    R: np.ndarray
    Vt: np.ndarray | None
    neg_w: int
    Rh: np.ndarray
    PKtz: np.ndarray
    ldet_XWX_plus_S: float
    beta_full: np.ndarray
    rS_work: np.ndarray
    deviance_hessian_half: np.ndarray


def _pivoted_economic_qr(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economic column-pivoted QR through SciPy's supported interface."""
    matrix = np.asarray(matrix, dtype=np.float64)
    q_mat, r_mat, pivot = qr(
        matrix,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    return (
        np.asarray(q_mat, dtype=np.float64),
        np.asarray(r_mat, dtype=np.float64),
        np.asarray(pivot, dtype=np.int64),
    )


def _undrop_rows_vec(
    v: np.ndarray, n_rows_full: int, drop_sorted: np.ndarray
) -> np.ndarray:
    """Mirror ``mgcv/src/gdi.c::undrop_rows()`` for a single packed column."""
    restored = restore_dropped_rows(
        np.asarray(v, dtype=np.float64).reshape(-1, 1),
        int(n_rows_full),
        np.asarray(drop_sorted, dtype=int),
    )
    return np.asarray(restored[:, 0], dtype=np.float64)


def _stacked_penalized_ls_nonneg_solution(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    penalty_sqrt: np.ndarray,
    penalty_rank_rows: np.ndarray,
    rank_tol: float,
) -> _StackedPlsNonnegOutcome:
    """
    Core non-negative-weight PLS solve mirroring ``mgcv/src/gdi.c::pls_fit1()``.

    The coefficient path follows the serial `mgcv_pqr` / `mgcv_qr` control flow
    directly, including the `eta` reconstruction and `use_wy` fallback test.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n_obs, n_coef_total = X.shape
    if z.shape[0] != n_obs or w.shape[0] != n_obs:
        raise ValueError("shape mismatch between X rows, z, and w.")

    penalty_sqrt = np.asarray(penalty_sqrt, dtype=np.float64)
    penalty_rank_rows = np.asarray(penalty_rank_rows, dtype=np.float64)
    n_penalty_rows = int(penalty_sqrt.shape[0])
    n_rank_penalty_rows = int(penalty_rank_rows.shape[0])
    if penalty_sqrt.ndim != 2 or penalty_sqrt.shape[1] != n_coef_total:
        raise ValueError("penalty_sqrt must have shape (r, ncol(X)).")
    if penalty_rank_rows.ndim != 2 or penalty_rank_rows.shape[1] != n_coef_total:
        raise ValueError("penalty_rank_rows must have shape (r, ncol(X)).")

    raw_w = np.sqrt(np.abs(w))
    neg_weight_mask = np.asarray(w < 0.0, dtype=bool)
    weighted_X = raw_w[:, None] * X
    n_wx_econ = min(n_obs, n_coef_total)
    n_augmented_rows = n_wx_econ + n_penalty_rows

    q_weighted, r_weighted, pivot_wx = _pivoted_economic_qr(weighted_X)
    r_weighted_natural = permute_columns(r_weighted, pivot_wx, reverse=True)

    frob_rw = float(np.linalg.norm(r_weighted_natural, ord="fro"))
    frob_es = float(np.linalg.norm(penalty_rank_rows, ord="fro"))
    if frob_rw <= 0.0:
        frob_rw = 1.0
    if frob_es <= 0.0:
        frob_es = 1.0

    n_stack_rows = n_wx_econ + n_rank_penalty_rows
    rank_stack = np.zeros((n_stack_rows, n_coef_total), dtype=np.float64)
    rank_stack[:n_wx_econ, :] = r_weighted_natural / frob_rw
    rank_stack[n_wx_econ:, :] = penalty_rank_rows / frob_es

    _q_rank, r_rank, pivot_rank = _pivoted_economic_qr(rank_stack)
    system_rank = min(n_coef_total, n_stack_rows)
    rcond = upper_triangular_condition_indicator(r_rank, system_rank)
    while system_rank > 0 and rank_tol * rcond > 1.0:
        system_rank -= 1
        rcond = upper_triangular_condition_indicator(r_rank, system_rank)

    system_rank = min(system_rank, n_augmented_rows)
    if n_coef_total > system_rank:
        dropped_column_indices = np.sort(
            np.asarray(
                [int(pivot_rank[i]) for i in range(system_rank, n_coef_total)],
                dtype=int,
            )
        )
    else:
        dropped_column_indices = np.zeros((0,), dtype=int)

    r_weighted_kept = drop_columns_dense(r_weighted_natural, dropped_column_indices)
    penalty_sqrt_kept = drop_columns_dense(penalty_sqrt, dropped_column_indices)
    drop_set = {int(d) for d in dropped_column_indices}
    kept_original_indices = [k for k in range(n_coef_total) if k not in drop_set]

    augmented_r = np.zeros((n_augmented_rows, system_rank), dtype=np.float64)
    augmented_r[:n_wx_econ, :] = r_weighted_kept
    augmented_r[n_wx_econ:, :] = penalty_sqrt_kept

    # Use a fresh pivoted QR. Reusing mgcv's raw JPVT work buffer is a
    # platform-specific LAPACK representation detail, not a fitted-model
    # invariant.
    q_augmented, r_augmented, pivot_aug = _pivoted_economic_qr(augmented_r)
    pivot_aug = np.asarray(pivot_aug[:system_rank], dtype=np.int64)
    upper_r_final = np.triu(
        np.asarray(r_augmented[:system_rank, :system_rank], dtype=np.float64)
    )

    r_weighted_pivoted = permute_columns(
        r_weighted_kept, pivot_aug, reverse=False
    )
    # gdi.c::gdiPK() obtains this from the unpenalized weighted-design R
    # factor, after the final rank pivot. The matrix product is the behavioral
    # quantity; BLAS-specific lower-triangle accumulation order is not.
    deviance_hessian_half = np.asarray(
        r_weighted_pivoted.T @ r_weighted_pivoted,
        dtype=np.float64,
    )

    signed_correction = None
    covariance_rank_root = None
    log_det_correction = 0.0
    if np.any(neg_weight_mask) and system_rank > 0:
        q1_matrix = np.asarray(
            q_weighted @ q_augmented[:n_wx_econ, :system_rank],
            dtype=np.float64,
        )
        (
            signed_correction,
            vt_scaled,
            _rh_left,
            log_det_correction,
            hessian_correction_basis,
        ) = _signed_weight_rank_correction(
            q1_matrix[neg_weight_mask, :],
            rank_tol=rank_tol,
        )
        hessian_correction_root = np.asarray(
            hessian_correction_basis @ upper_r_final,
            dtype=np.float64,
        )
        deviance_hessian_half -= 2.0 * np.asarray(
            hessian_correction_root.T @ hessian_correction_root,
            dtype=np.float64,
        )
        covariance_rank_root = solve_triangular(
            upper_r_final,
            vt_scaled.T,
            lower=False,
            check_finite=False,
        )

    nz = max(n_obs, n_augmented_rows)
    z_buf = np.zeros(nz, dtype=np.float64)
    z_buf[:n_obs] = z * raw_w
    z_buf[:n_obs][neg_weight_mask] *= -1.0
    qrz_rank_raw = np.zeros(system_rank, dtype=np.float64)
    # `pls_fit1` requires zero-weight observations to have been removed before
    # entering its fast eta-reconstruction path.  Its documented alternative
    # for reciprocal/zero working weights is the direct X'Wz path (`use_wy`).
    # Our callers retain the original row layout, so select that upstream path
    # whenever a zero weight is present instead of dividing by sqrt(w).
    use_wy = bool(np.any(raw_w == 0.0))
    penalty_quadratic = 0.0
    eta = np.zeros(n_obs, dtype=np.float64)

    if not use_wy:
        z_qt = np.asarray(q_weighted.T @ z_buf[:n_obs], dtype=np.float64)
        z_buf.fill(0.0)
        z_buf[:n_wx_econ] = z_qt
        z_q1t = np.asarray(
            q_augmented.T @ z_buf[:n_augmented_rows],
            dtype=np.float64,
        )
        z_buf.fill(0.0)
        z_buf[:system_rank] = z_q1t
        qrz_rank_raw = np.asarray(z_q1t, dtype=np.float64).copy()
        y_rank = qrz_rank_raw.copy()
        if signed_correction is not None:
            y_rank = np.asarray(signed_correction @ y_rank, dtype=np.float64)

        z_q1 = np.asarray(q_augmented @ y_rank, dtype=np.float64)
        penalty_quadratic = float(np.sum(z_q1[n_wx_econ:n_augmented_rows] ** 2))
        z_buf.fill(0.0)
        z_buf[:system_rank] = z_q1[:system_rank]
        weighted_eta = np.asarray(
            q_weighted @ z_buf[:n_wx_econ],
            dtype=np.float64,
        )
        eta = np.asarray(weighted_eta / raw_w, dtype=np.float64)
    else:
        y_rank = np.zeros(system_rank, dtype=np.float64)

    xwz = X.T @ (w * z)
    xwz_kept = _drop_rows_vec(xwz, dropped_column_indices)
    xwz_pivoted = np.asarray(xwz_kept[pivot_aug], dtype=np.float64)

    if not use_wy:
        recon_error_sq = 0.0
        target_norm_sq = 0.0
        for i in range(system_rank):
            xx = 0.0
            for j in range(i + 1):
                xx += upper_r_final[j, i] * qrz_rank_raw[j]
            xx -= xwz_pivoted[i]
            recon_error_sq += xx * xx
            target_norm_sq += xwz_pivoted[i] * xwz_pivoted[i]
        if recon_error_sq > rank_tol * target_norm_sq:
            use_wy = True

    z_rank = np.zeros(system_rank, dtype=np.float64)
    if use_wy:
        for k in range(system_rank):
            xx = 0.0
            for j in range(k):
                xx += upper_r_final[j, k] * z_rank[j]
            z_rank[k] = (
                xwz_pivoted[k] / upper_r_final[k, k]
                if k == 0
                else (xwz_pivoted[k] - xx) / upper_r_final[k, k]
            )
        y_rank = z_rank.copy()
        if signed_correction is not None:
            y_rank = np.asarray(signed_correction @ y_rank, dtype=np.float64)

    for k in range(system_rank - 1, -1, -1):
        xx = 0.0
        for j in range(k + 1, system_rank):
            xx += upper_r_final[k, j] * z_rank[j]
        z_rank[k] = (y_rank[k] - xx) / upper_r_final[k, k]

    coef_kept = np.zeros(system_rank, dtype=np.float64)
    coef_kept[pivot_aug] = z_rank
    coef_full = _undrop_rows_vec(coef_kept, n_coef_total, dropped_column_indices)

    if use_wy:
        eta = np.asarray(X @ coef_full, dtype=np.float64)
        pen_vec = penalty_sqrt @ coef_full
        penalty_quadratic = float(pen_vec @ pen_vec)

    return _StackedPlsNonnegOutcome(
        coef_full=coef_full,
        eta=eta,
        penalty_quadratic=penalty_quadratic,
        penalty_sqrt=penalty_sqrt,
        weighted_X=weighted_X,
        system_rank=int(system_rank),
        upper_r_final=upper_r_final,
        q_weighted=q_weighted,
        q_augmented=q_augmented,
        pivot_aug=pivot_aug,
        n_wx_econ=int(n_wx_econ),
        kept_original_indices=kept_original_indices,
        dropped_column_indices=dropped_column_indices,
        covariance_rank_root=(
            None
            if covariance_rank_root is None
            else np.asarray(covariance_rank_root, dtype=np.float64)
        ),
        log_det_correction=float(log_det_correction),
        deviance_hessian_half=np.asarray(
            deviance_hessian_half, dtype=np.float64
        ).copy(),
    )


def _validate_nonnegative_qr_inputs(X, z, w, penalty_sqrt_E, penalty_rank_Es, rS):
    X = np.asarray(X, dtype=np.float64, order="C")
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n, q = X.shape
    if z.shape[0] != n or w.shape[0] != n:
        raise ValueError("z and w must have length n (rows of X).")
    if np.any(~np.isfinite(w)):
        raise ValueError("w must be finite.")
    E = np.asarray(penalty_sqrt_E, dtype=np.float64, order="C")
    Es = np.asarray(penalty_rank_Es, dtype=np.float64, order="C")
    rS = np.asarray(rS, dtype=np.float64, order="C")
    if E.ndim != 2 or Es.ndim != 2:
        raise ValueError("penalty_sqrt_E and penalty_rank_Es must be 2-D.")
    enrow = int(E.shape[0])
    if E.shape[1] != q or Es.shape[1] != q or Es.shape[0] != enrow:
        raise ValueError("E and Es must have q columns matching X.")
    if rS.shape[0] != q:
        raise ValueError("rS must have q rows.")
    return X, z, w, E, Es, rS


def _signed_weight_rank_correction(
    q1_negative_rows: np.ndarray,
    *,
    rank_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Mirror `mgcv/src/gdi.c::pls_fit1()` signed-weight SVD correction.

    Returns coefficient correction `V (I - 2 D^2)^-1 V'`, covariance/root
    right factor `(I - 2 D^2)^-1/2 V'`, left factor `(I - 2 D^2)^1/2 V'`
    for `Rh`, the determinant correction, and ``D V'`` for gdiPK's signed
    deviance-Hessian correction.
    """
    q1_negative_rows = np.asarray(q1_negative_rows, dtype=np.float64)
    if q1_negative_rows.ndim != 2:
        raise ValueError("q1_negative_rows must be 2-D.")
    rank = int(q1_negative_rows.shape[1])
    if rank == 0:
        z = np.zeros((0, 0), dtype=np.float64)
        return z, z, z, 0.0, z

    n_neg = int(q1_negative_rows.shape[0])
    k = max(n_neg, rank + 1)
    iq = np.zeros((k, rank), dtype=np.float64)
    if n_neg:
        iq[:n_neg, :] = q1_negative_rows

    _u, sing_vals, vt = np.linalg.svd(iq, full_matrices=False)
    delta = 1.0 - 2.0 * sing_vals * sing_vals
    if np.any(delta < -rank_tol):
        raise np.linalg.LinAlgError(
            "signed-weight stacked QR system is not positive definite."
        )

    log_det_correction = 0.0
    inv_sqrt = np.zeros(rank, dtype=np.float64)
    delta_pos = np.zeros(rank, dtype=np.float64)
    for i, di in enumerate(delta):
        if di > 0.0:
            log_det_correction += float(np.log(di))
            inv_sqrt[i] = float(1.0 / np.sqrt(di))
            delta_pos[i] = float(di)

    vt_scaled = inv_sqrt[:, None] * np.asarray(vt, dtype=np.float64)
    rh_left = delta_pos[:, None] * vt_scaled
    correction = np.asarray(vt_scaled.T @ vt_scaled, dtype=np.float64)
    hessian_correction_basis = np.asarray(
        sing_vals[:, None] * np.asarray(vt, dtype=np.float64),
        dtype=np.float64,
    )
    return (
        correction,
        vt_scaled,
        rh_left,
        float(log_det_correction),
        hessian_correction_basis,
    )


def _scatter_rank_root_to_full(rank_root, *, pivot1, kept_original_indices, q_total):
    root_full = _scatter_pivoted_rank_matrix_to_full(
        np.asarray(rank_root, dtype=np.float64),
        kept_original_indices=kept_original_indices,
        pivot1=np.asarray(pivot1, dtype=np.int64),
        q_total=int(q_total),
    )
    cov = np.asarray(root_full @ root_full.T, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return root_full, cov


def _compute_nonnegative_pk_rhs(Rh, R_nr_rank, K, X, z, w, drop, pivot1):
    P = solve_triangular(Rh, np.eye(Rh.shape[0], dtype=np.float64), lower=False)
    raw = np.sqrt(np.maximum(w, 0.0))
    zz = z * raw
    ktz = K.T @ zz
    q1tz = K.T @ zz
    wz = w * z
    X_drop = drop_columns_dense(X, drop)
    X_piv = permute_columns(X_drop, pivot1, reverse=False)
    xwz = X_piv.T @ wz

    norm1 = 0.0
    norm2 = 0.0
    rank = Rh.shape[0]
    for i in range(rank):
        s = 0.0
        for j in range(i + 1):
            s += R_nr_rank[j, i] * q1tz[j]
        diff = s - xwz[i]
        norm1 += diff * diff
        norm2 += xwz[i] * xwz[i]
    return P, ktz, xwz, norm1, norm2


def _scatter_nonnegative_coefficients(PKtz, kept, pivot1, q):
    coef_nat = np.zeros(PKtz.shape[0], dtype=np.float64)
    coef_nat[pivot1] = PKtz
    beta_full = np.zeros(q, dtype=np.float64)
    for j, orig_col in enumerate(kept):
        beta_full[orig_col] = coef_nat[j]
    return beta_full


def _scatter_pivoted_rank_matrix_to_full(
    pivoted_rank_matrix: np.ndarray,
    kept_original_indices: list[int] | tuple[int, ...] | np.ndarray,
    pivot1: np.ndarray,
    q_total: int,
) -> np.ndarray:
    """
    Undo stacked-QR drop/pivot bookkeeping on a rank-space matrix.

    Mirrors `mgcv/src/gdi.c` post-processing: the final `P = R^{-1}` lives in the
    pivoted reduced parameterization, so covariance roots need both the final
    column unpivot and the dropped-column reinsertion to get back to the full
    coefficient space.
    """
    pivoted_rank_matrix = np.asarray(pivoted_rank_matrix, dtype=np.float64)
    pivot1 = np.asarray(pivot1, dtype=np.int64).ravel()
    kept = np.asarray(kept_original_indices, dtype=np.int64).ravel()

    if pivoted_rank_matrix.ndim != 2:
        raise ValueError("pivoted_rank_matrix must be 2-D.")
    if pivoted_rank_matrix.shape[0] != pivot1.size:
        raise ValueError("pivoted_rank_matrix row count must match final pivot length.")
    if kept.size != pivot1.size:
        raise ValueError(
            "kept_original_indices length must match final stacked-QR rank."
        )

    unpivoted_kept = np.zeros(
        (kept.size, pivoted_rank_matrix.shape[1]), dtype=np.float64
    )
    unpivoted_kept[pivot1, :] = pivoted_rank_matrix

    full = np.zeros((int(q_total), pivoted_rank_matrix.shape[1]), dtype=np.float64)
    full[kept, :] = unpivoted_kept
    return full


def stacked_qr_covariance_from_factor(
    upper_r_final: np.ndarray,
    *,
    pivot1: np.ndarray,
    kept_original_indices: list[int] | tuple[int, ...] | np.ndarray,
    q_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank-aware covariance root and covariance for stacked-QR Gaussian solves.

    Mirrors `mgcv::magic.post.proc()` / `gdiPK()` logic: if the penalized system
    is rank deficient, covariance is formed from the reduced `R^{-1}` root in the
    identified parameter space and then scattered back to the full coefficient
    layout, rather than by inverting the singular full `X'WX + S`.
    """
    upper_r_final = np.asarray(upper_r_final, dtype=np.float64)
    if upper_r_final.ndim != 2 or upper_r_final.shape[0] != upper_r_final.shape[1]:
        raise ValueError("upper_r_final must be square.")
    rank = int(upper_r_final.shape[0])
    if rank == 0:
        root = np.zeros((int(q_total), 0), dtype=np.float64)
        cov = np.zeros((int(q_total), int(q_total)), dtype=np.float64)
        return root, cov

    rank_root = solve_triangular(
        upper_r_final,
        np.eye(rank, dtype=np.float64),
        lower=False,
        check_finite=False,
    )
    root_full = _scatter_pivoted_rank_matrix_to_full(
        rank_root,
        kept_original_indices=kept_original_indices,
        pivot1=pivot1,
        q_total=int(q_total),
    )
    cov = np.asarray(root_full @ root_full.T, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return root_full, cov


def build_penalized_qr_state_nonnegative(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    penalty_sqrt_E: np.ndarray,
    penalty_rank_Es: np.ndarray,
    rS: np.ndarray,
    rank_tol: float,
    reml: bool,
) -> NonnegativePenalizedQRState:
    X, z, w, E, Es, rS = _validate_nonnegative_qr_inputs(
        X, z, w, penalty_sqrt_E, penalty_rank_Es, rS
    )
    q = X.shape[1]

    out = _stacked_penalized_ls_nonneg_solution(
        X,
        z,
        w,
        penalty_sqrt=E,
        penalty_rank_rows=Es,
        rank_tol=rank_tol,
    )

    rank = int(out.system_rank)
    rr = int(out.n_wx_econ)
    drop = np.asarray(out.dropped_column_indices, dtype=int)
    n_drop = int(drop.size)
    pivot1 = np.asarray(out.pivot_aug, dtype=int).ravel()
    if pivot1.size != rank:
        raise RuntimeError("internal: pivot_aug length must equal system rank.")

    kept = tuple(int(k) for k in out.kept_original_indices)
    q1_matrix = np.asarray(
        out.q_weighted @ out.q_augmented[:rr, :rank],
        dtype=np.float64,
    )
    R = np.triu(np.asarray(out.upper_r_final, dtype=np.float64))
    raw_w = np.sqrt(np.abs(w))
    neg_weight_mask = np.asarray(w < 0.0, dtype=bool)
    X_drop = drop_columns_dense(X, drop) if n_drop else X.copy()
    X_piv = permute_columns(X_drop, pivot1, reverse=False)

    Vt = None
    if np.any(neg_weight_mask):
        (
            _,
            vt_scaled,
            rh_left,
            log_det_correction,
            _hessian_correction_basis,
        ) = _signed_weight_rank_correction(
            q1_matrix[neg_weight_mask, :], rank_tol=rank_tol
        )
        if rank == 0:
            P = np.empty((0, 0), dtype=np.float64)
        else:
            P = solve_triangular(
                R,
                vt_scaled.T,
                lower=False,
                check_finite=False,
            )
        K = np.asarray((raw_w[:, None] * X_piv) @ P, dtype=np.float64)
        Rh = np.asarray(rh_left @ R, dtype=np.float64)
        Vt = np.asarray(vt_scaled, dtype=np.float64)

        zz = z * raw_w
        zz[neg_weight_mask] *= -1.0
        q1tz = np.asarray(q1_matrix.T @ zz, dtype=np.float64)
        ktz = np.asarray(K.T @ zz, dtype=np.float64)
        xwz = np.asarray(X_piv.T @ (w * z), dtype=np.float64)

        norm1 = 0.0
        norm2 = 0.0
        for i in range(rank):
            s = 0.0
            for j in range(i + 1):
                s += R[j, i] * q1tz[j]
            diff = s - xwz[i]
            norm1 += diff * diff
            norm2 += xwz[i] * xwz[i]

        if norm1 > rank_tol * norm2:
            tmp = solve_triangular(R, xwz, lower=False, trans="T", check_finite=False)
            PKtz = np.asarray(P @ (vt_scaled @ tmp), dtype=np.float64)
        else:
            PKtz = np.asarray(P @ ktz, dtype=np.float64)

        if reml:
            d = np.abs(np.diag(R))
            ldet = float(
                2.0 * np.sum(np.log(np.maximum(d, np.finfo(np.float64).tiny)))
                + log_det_correction
            )
        else:
            ldet = 0.0
    else:
        K = q1_matrix
        Rh = R
        P, ktz, xwz, norm1, norm2 = _compute_nonnegative_pk_rhs(
            Rh, R, K, X, z, w, drop, pivot1
        )
        if norm1 > rank_tol * norm2:
            tmp = solve_triangular(Rh, xwz, lower=False, trans="T")
            PKtz = solve_triangular(Rh, tmp, lower=False)
        else:
            PKtz = solve_triangular(Rh, ktz, lower=False)

        if reml:
            d = np.abs(np.diag(Rh))
            ldet = float(2.0 * np.sum(np.log(np.maximum(d, np.finfo(np.float64).tiny))))
        else:
            ldet = 0.0

    beta_full = _scatter_nonnegative_coefficients(PKtz, kept, pivot1, q)
    rS_drop = drop_rows_dense(rS, drop) if n_drop else rS.copy()
    rS_work = permute_rows(rS_drop, pivot1, reverse=False)

    return NonnegativePenalizedQRState(
        rank=rank,
        n_drop=n_drop,
        drop=drop.copy(),
        pivot1=pivot1.copy(),
        kept_original_indices=kept,
        P=P,
        K=K,
        R=R,
        Vt=Vt,
        neg_w=int(np.sum(neg_weight_mask)),
        Rh=Rh,
        PKtz=np.asarray(PKtz, dtype=np.float64).ravel().copy(),
        ldet_XWX_plus_S=ldet,
        beta_full=beta_full,
        rS_work=rS_work,
        deviance_hessian_half=np.asarray(
            out.deviance_hessian_half, dtype=np.float64
        ).copy(),
    )


def _drop_rows_vec(v: np.ndarray, drop_sorted: np.ndarray) -> np.ndarray:
    """``mgcv`` ``drop_rows`` for a single column."""
    if drop_sorted.size == 0:
        return np.asarray(v, dtype=np.float64).copy()
    q = int(v.shape[0])
    mask = np.ones(q, dtype=bool)
    mask[drop_sorted.astype(int)] = False
    out: np.ndarray = np.array(v, dtype=np.float64, copy=True)[mask]
    return out


def penalty_sqrt_rows(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Penalty square-root that preserves diagonal structure when ``P`` is diagonal.

    When ``P`` is (numerically) diagonal, an eigendecomposition can rotate a
    repeated eigenspace before the upstream-style triangular solve. This
    function uses one identity row per positive diagonal entry instead.
    Falls back to eigendecomposition when ``P`` has off-diagonal entries.
    """
    P = np.asarray(P, dtype=np.float64)
    P = symmetrize_matrix(P)
    q = int(P.shape[0])
    d = np.diag(P)
    off = P - np.diag(d)
    scale = float(np.max(np.abs(P))) if q else 1.0
    if scale <= 0.0:
        scale = 1.0
    if not np.allclose(off, 0.0, atol=1e-14 * max(scale, 1.0), rtol=0.0):
        w, V = symmetric_eigh(P)
        mask = w > max(np.max(w) * 1e-15, LOG_GUARD_MIN)
        if not np.any(mask):
            return np.zeros((0, P.shape[0])), np.zeros((0, P.shape[0]))
        wl = np.sqrt(np.maximum(w[mask], 0.0))
        Vp = V[:, mask]
        E = wl[:, None] * Vp.T  # r × q
        row_norms = np.linalg.norm(E, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, LOG_GUARD_MIN)
        Es = E / row_norms
        return E, Es
    rows: list[np.ndarray] = []
    thr = max(np.max(d) * 1e-15, LOG_GUARD_MIN)
    for j in range(q):
        if d[j] > thr:
            r = np.zeros(q, dtype=np.float64)
            r[j] = float(np.sqrt(max(d[j], 0.0)))
            rows.append(r)
    if not rows:
        return np.zeros((0, q)), np.zeros((0, q))
    E = np.vstack(rows)
    row_norms = np.linalg.norm(E, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, LOG_GUARD_MIN)
    Es = E / row_norms
    return E, Es


def pls_fit1_nonneg_w(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    wy: np.ndarray,
    *,
    penalty_sqrt_E: np.ndarray,
    penalty_rank_Es: np.ndarray,
    rank_tol: float = STACKED_QR_RANK_TOLERANCE,
) -> tuple[np.ndarray, float]:
    """
    One penalized weighted least-squares step (non-negative weights only).

    Minimises ``||W^{1/2}(X beta - z)||^2 + beta' S beta`` where ``S = E'E``,
    using the stacked pivoted QR algorithm.  This is the inner solve called once
    per IRLS iteration.

    Parameters
    ----------
    X, z, w, wy
        Rows already filtered to informative observations (``prior_weight > 0``).
        ``wy`` must equal ``w * z`` elementwise.
    penalty_sqrt_E
        Penalty square-root factor ``E`` with ``E.T @ E = S`` (total penalty).
    penalty_rank_Es
        Balanced penalty rows used for numerical rank detection only.
    Returns
    -------
    coef_full, penalty_quadratic

    Notes
    -----
    Negative Newton working weights are not supported; the caller should fall back
    to Fisher scoring when Newton weights are non-positive.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    wy = np.asarray(wy, dtype=np.float64).ravel()
    n_obs, n_coef_total = X.shape
    if z.shape[0] != n_obs or w.shape[0] != n_obs or wy.shape[0] != n_obs:
        raise ValueError("shape mismatch among X rows, z, w, and wy.")
    exp_wy = w * z
    if not np.allclose(
        wy, exp_wy, rtol=0.0, atol=float(np.finfo(np.float64).eps * 8)
    ):
        raise ValueError("wy must equal w * z elementwise (mgcv pls_fit1 contract).")
    out = _stacked_penalized_ls_nonneg_solution(
        X,
        z,
        w,
        penalty_sqrt=np.asarray(penalty_sqrt_E, dtype=np.float64),
        penalty_rank_rows=np.asarray(penalty_rank_Es, dtype=np.float64),
        rank_tol=rank_tol,
    )
    return out.coef_full, out.penalty_quadratic


def solve_gaussian_penalized_ls_stacked_qr(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    *,
    penalty_blocks: Iterable[Any] | None = None,
    penalty_sqrt_E: np.ndarray | None = None,
    penalty_rank_rows: np.ndarray | None = None,
    fit_intercept: bool = True,
    n_coef: int | None = None,
    rank_tol: float = STACKED_QR_RANK_TOLERANCE,
) -> dict:
    """
    Full Gaussian penalized least-squares solve returning EDF and covariance matrices.

    Calls the stacked pivoted QR solver and then reconstructs the matrices needed
    for EDF computation and post-fit diagnostics.

    Parameters
    ----------
    penalty_sqrt_E
        Exact current-smoothing-parameter penalty root. Model fitting paths
        pass the ``Sr`` produced by ``mgcv::gam.reparam``.
    penalty_blocks, fit_intercept, n_coef
        When ``penalty_blocks`` is provided (along with ``n_coef``), rank detection
        uses :func:`balanced_penalty_template_sqrt_for_rank`.  Omit these to fall back
        to row-normalised ``sqrt(P)``, which is less stable for mixed-scale penalties.
    rank_tol
        Condition threshold for the rank-reveal step; default is
        :data:`STACKED_QR_RANK_TOLERANCE`.

    Returns
    -------
    dict
        Keys include ``coef_full``, ``covariance_root`` (q×rank `rV`
        analogue), ``A_inv`` (rank-aware covariance / pseudoinverse analogue),
        ``coef_hat_matrix`` (the F matrix for EDF computation),
        ``log_det_XtWX_plus_penalty``, ``penalized_system_rank``,
        ``dropped_column_indices``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n_obs, n_coef_total = X.shape
    if y.shape[0] != n_obs or w.shape[0] != n_obs:
        raise ValueError("shape mismatch between X, y, and w.")

    P = np.asarray(P, dtype=np.float64)
    if penalty_sqrt_E is None:
        penalty_sqrt, penalty_rank_template = penalty_sqrt_rows(P)
    else:
        penalty_sqrt = np.asarray(penalty_sqrt_E, dtype=np.float64)
        if penalty_sqrt.ndim != 2 or penalty_sqrt.shape[1] != n_coef_total:
            raise ValueError(
                "penalty_sqrt_E must be a two-dimensional current-SP root "
                f"with {n_coef_total} columns."
            )
        penalty_rank_template = np.asarray(penalty_sqrt, dtype=np.float64)

    if penalty_rank_rows is not None:
        penalty_rank_rows = np.asarray(penalty_rank_rows, dtype=np.float64)
    elif penalty_blocks is not None and n_coef is not None:
        penalty_rank_rows = balanced_penalty_template_sqrt_for_rank(
            penalty_blocks, fit_intercept=fit_intercept, n_coef=int(n_coef)
        )
    else:
        penalty_rank_rows = np.asarray(penalty_rank_template, dtype=np.float64)

    outcome = _stacked_penalized_ls_nonneg_solution(
        X,
        y,
        w,
        penalty_sqrt=penalty_sqrt,
        penalty_rank_rows=penalty_rank_rows,
        rank_tol=rank_tol,
    )
    coef_full = outcome.coef_full
    eta = np.asarray(outcome.eta, dtype=np.float64)
    weighted_X = outcome.weighted_X
    penalty_sqrt = outcome.penalty_sqrt
    system_rank = outcome.system_rank
    upper_r_final = outcome.upper_r_final
    pivot_aug = outcome.pivot_aug
    kept_original_indices = outcome.kept_original_indices
    dropped_column_indices = outcome.dropped_column_indices
    penalty_quadratic = outcome.penalty_quadratic
    covariance_rank_root = outcome.covariance_rank_root
    log_det_correction = float(outcome.log_det_correction)

    XtWX = X.T @ (w[:, None] * X)
    A = XtWX + P
    if covariance_rank_root is None:
        covariance_root, A_inv = stacked_qr_covariance_from_factor(
            upper_r_final,
            pivot1=pivot_aug,
            kept_original_indices=kept_original_indices,
            q_total=n_coef_total,
        )
    else:
        covariance_root, A_inv = _scatter_rank_root_to_full(
            covariance_rank_root,
            pivot1=pivot_aug,
            kept_original_indices=kept_original_indices,
            q_total=n_coef_total,
        )
    diag_r = np.abs(np.diag(upper_r_final))
    log_det_XtWX_plus_penalty = (
        2.0 * float(np.sum(np.log(np.maximum(diag_r, np.finfo(np.float64).tiny))))
        + log_det_correction
    )

    # Mirror mgcv post-processing: EDF uses the rank-aware covariance assembled
    # from the reduced QR factor, not a dense inverse of singular X'WX + S.
    coef_hat_matrix = A_inv @ XtWX

    return {
        "coef_full": coef_full,
        "eta": eta,
        "penalty_quadratic": penalty_quadratic,
        "XtWX": XtWX,
        "A": A,
        "A_inv": A_inv,
        "WX_sqrt": weighted_X,
        "E": penalty_sqrt,
        "covariance_root": covariance_root,
        "log_det_XtWX_plus_penalty": log_det_XtWX_plus_penalty,
        "penalized_system_rank": int(system_rank),
        "coef_hat_matrix": coef_hat_matrix,
        "dropped_column_indices": dropped_column_indices,
    }
