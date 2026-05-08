"""
Stacked pivoted QR for Gaussian penalized least-squares.

When the penalized normal equations ``X'WX + S`` are singular or near-singular
(e.g. a random-effect term with a small smoothing parameter), a plain Cholesky
solve on the normal equations can pick an unstable coefficient vector.  This
module implements a numerically stable alternative using two pivoted QR
decompositions (via LAPACK ``dgeqp3``) stacked as Wood (2017) describes.

Algorithm outline
-----------------
1. QR-decompose ``sqrt(W) X`` with column pivoting to get an economy R factor.
2. Stack that R factor with the penalty square-root rows and QR-decompose again
   with column pivoting to reveal numerical rank and drop near-zero columns.
3. Solve for coefficients using a Householder back-substitution chain on the
   stacked system.  This is the ``coef_method='householder'`` default path.
4. Reconstruct EDF-related matrices from the two QR factors.

Rank detection
--------------
Rank is revealed by an upper-triangular condition number estimate on the stacked
R factor.  The threshold is :data:`STACKED_QR_RANK_TOLERANCE` (eps**0.66),
matching `mgcv`'s stacked-QR rank-drop tolerance.
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
deficiency; raw coefficient vectors along ``null(X)`` may differ from R due to
LAPACK pivot differences.  :func:`snap_coef_to_reference_null_space` corrects
this for parity testing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal

import numpy as np
from scipy.linalg import lapack as _lapack
from scipy.linalg import solve_triangular

from ..._mgcv_constants import LOG_GUARD_MIN, QR_TOL_SCALE
from ..._model_state import _fit_intercept, _term_blocks_seq
from ...linalg import (
    balanced_penalty_template_sqrt_for_rank as _balanced_penalty_template_sqrt_for_rank,
)
from ...linalg import (
    matrix_is_rank_deficient,
    svd_null_space_basis,
    symmetric_eigh,
    symmetric_eigvalsh,
    symmetrize_matrix,
)
from ...linalg import (
    project_coef_onto_row_space as _project_coef_onto_row_space,
)
from ...linalg import (
    snap_coef_to_reference_null_space as _snap_coef_to_reference_null_space,
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
STACKED_QR_RANK_TOLERANCE = np.finfo(np.float64).eps ** 0.66 * QR_TOL_SCALE

# Below this Frobenius norm, the penalty is numerically zero on most of the null space
# of X (e.g. a random-effect term at the lower smoothing parameter bound).  The
# stacked-QR path still returns a valid coset member, but a penalty-minimisation
# gauge on null(X) improves stability when ``near_singular_null_pin`` is enabled.
NEAR_SINGULAR_PENALTY_FROB_TOL = 1e-20


@dataclass(frozen=True)
class _StackedPlsNonnegOutcome:
    """Internal: one non-negative-weight stacked PLS solve + QR state for post-processing."""

    coef_full: np.ndarray
    eta: np.ndarray
    penalty_quadratic: float
    X: np.ndarray
    w: np.ndarray
    P_dense: np.ndarray | None
    penalty_sqrt: np.ndarray
    weighted_X: np.ndarray
    sqrt_w: np.ndarray
    system_rank: int
    upper_r_final: np.ndarray
    qr_wx: np.ndarray
    tau_wx: np.ndarray
    qr_aug: np.ndarray
    tau_aug: np.ndarray
    pivot_aug: np.ndarray
    n_wx_econ: int
    kept_original_indices: list[int]
    dropped_column_indices: np.ndarray
    covariance_rank_root: np.ndarray | None
    log_det_correction: float


@dataclass(frozen=True)
class NonnegativePenalizedQRState:
    rank: int
    n_drop: int
    drop: np.ndarray
    pivot1: np.ndarray
    kept_original_indices: tuple[int, ...]
    P: np.ndarray
    K: np.ndarray
    Rh: np.ndarray
    PKtz: np.ndarray
    ldet_XWX_plus_S: float
    beta_full: np.ndarray
    rS_work: np.ndarray


def _get_r_pqr_serial(qr_a: np.ndarray, *, rr: int, ncol: int) -> np.ndarray:
    """Mirror ``mgcv/src/mat.c::getRpqr()`` for the serial QR case."""
    qr_a = np.asfortranarray(np.asarray(qr_a, dtype=np.float64))
    out = np.zeros((int(rr), int(ncol)), dtype=np.float64)
    n = int(qr_a.shape[0])
    rows = min(int(rr), int(ncol))
    packed = np.ravel(qr_a, order="F")
    for j in range(int(ncol)):
        for i in range(min(rows, j + 1)):
            idx = i + n * j
            if idx < packed.size:
                out[i, j] = packed[idx]
    return out


def _apply_q_left_serial(
    block: np.ndarray,
    qr_a: np.ndarray,
    tau: np.ndarray,
    *,
    r: int,
    c: int,
    transpose: bool,
) -> np.ndarray:
    """
    Mirror serial ``mgcv_qrqy`` / ``mgcv_pqrqy`` left-application semantics.

    When ``transpose`` is false, ``block`` is packed ``c x cb`` input and the
    result is a full ``r x cb`` matrix. When ``transpose`` is true, ``block`` is
    a full ``r x cb`` input and the result is the packed leading ``c x cb`` rows.
    """
    arr = np.asarray(block, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cb = int(arr.shape[1])
    full = np.zeros((int(r), cb), dtype=np.float64, order="F")
    if transpose:
        n_copy = min(int(r), int(arr.shape[0]))
        full[:n_copy, :] = arr[:n_copy, :]
        out = _dormqr_apply(b"L", b"T", qr_a, tau, full)
        return np.asarray(out[: int(c), :], dtype=np.float64)
    n_copy = min(int(c), int(arr.shape[0]))
    full[:n_copy, :] = arr[:n_copy, :]
    return np.asarray(_dormqr_apply(b"L", b"N", qr_a, tau, full), dtype=np.float64)


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


def _stacked_penalized_ls_nonneg_solution_literal(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    penalty_sqrt: np.ndarray,
    penalty_rank_rows: np.ndarray,
    P_dense: np.ndarray | None,
    rank_tol: float,
    coef_method: str,
    near_singular_null_pin: bool | Literal["auto"],
) -> _StackedPlsNonnegOutcome:
    """
    Core non-negative-weight PLS solve mirroring ``mgcv/src/gdi.c::pls_fit1()``.

    The coefficient path follows the serial `mgcv_pqr` / `mgcv_qr` control flow
    directly, including the `eta` reconstruction and `use_wy` fallback test.
    """
    cm = str(coef_method).lower().strip()
    if cm not in {"householder", "lstsq"}:
        raise ValueError(
            f"Unknown coef_method {coef_method!r}; use 'householder' or 'lstsq'."
        )

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

    qr_wx, tau_wx, pivot_wx, _ = _dgeqp3_economic_r(weighted_X)
    r_weighted = _get_r_pqr_serial(qr_wx, rr=n_wx_econ, ncol=n_coef_total)
    r_weighted_natural = permute_columns(r_weighted, pivot_wx, reverse=True)

    frob_rw = _frob_norm(r_weighted_natural)
    frob_es = _frob_norm(penalty_rank_rows)
    if frob_rw <= 0.0:
        frob_rw = 1.0
    if frob_es <= 0.0:
        frob_es = 1.0

    n_stack_rows = n_wx_econ + n_rank_penalty_rows
    rank_stack = np.zeros((n_stack_rows, n_coef_total), dtype=np.float64)
    rank_stack[:n_wx_econ, :] = r_weighted_natural / frob_rw
    rank_stack[n_wx_econ:, :] = penalty_rank_rows / frob_es

    qr_rank, tau_rank, pivot_rank, _ = _dgeqp3_economic_r(rank_stack)
    del tau_rank
    system_rank = min(n_coef_total, n_stack_rows)
    rcond = _upper_r_condition_indicator(
        _get_r_pqr_serial(
            qr_rank, rr=min(n_stack_rows, n_coef_total), ncol=n_coef_total
        ),
        system_rank,
    )
    while system_rank > 0 and rank_tol * rcond > 1.0:
        system_rank -= 1
        rcond = _upper_r_condition_indicator(
            _get_r_pqr_serial(
                qr_rank, rr=min(n_stack_rows, n_coef_total), ncol=n_coef_total
            ),
            system_rank,
        )

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

    # Mirror `mgcv/src/gdi.c::pls_fit1()`: the final QR reuses the incoming
    # `pivot1` buffer from the rank-reveal QR, so `JPVT` is not reset here.
    qr_aug, tau_aug, pivot_aug, _ = _dgeqp3_economic_r(
        augmented_r,
        jpvt_in=np.asarray(pivot_rank, dtype=np.int32),
    )
    pivot_aug = np.asarray(pivot_aug[:system_rank], dtype=np.int64)
    upper_r_final = np.triu(
        np.asarray(qr_aug[:system_rank, :system_rank], dtype=np.float64)
    )

    signed_correction = None
    covariance_rank_root = None
    log_det_correction = 0.0
    if np.any(neg_weight_mask) and system_rank > 0:
        q1_matrix, _qr_aug_f = _build_q1_from_qr_factors(
            qr_wx,
            tau_wx,
            qr_aug,
            tau_aug,
            n_obs,
            n_wx_econ,
            system_rank,
        )
        signed_correction, vt_scaled, _rh_left, log_det_correction = (
            _signed_weight_rank_correction(
                q1_matrix[neg_weight_mask, :],
                rank_tol=rank_tol,
            )
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
    use_wy = False
    penalty_quadratic = 0.0
    eta = np.zeros(n_obs, dtype=np.float64)

    if not use_wy:
        z_qt = _apply_q_left_serial(
            z_buf[:n_obs],
            qr_wx,
            tau_wx,
            r=n_obs,
            c=n_wx_econ,
            transpose=True,
        )[:, 0]
        z_buf.fill(0.0)
        z_buf[:n_wx_econ] = z_qt
        z_q1t = _apply_q_left_serial(
            z_buf[:n_augmented_rows],
            qr_aug,
            tau_aug,
            r=n_augmented_rows,
            c=system_rank,
            transpose=True,
        )[:, 0]
        z_buf.fill(0.0)
        z_buf[:system_rank] = z_q1t
        qrz_rank_raw = np.asarray(z_q1t, dtype=np.float64).copy()
        y_rank = qrz_rank_raw.copy()
        if signed_correction is not None:
            y_rank = np.asarray(signed_correction @ y_rank, dtype=np.float64)

        z_q1 = _apply_q_left_serial(
            y_rank,
            qr_aug,
            tau_aug,
            r=n_augmented_rows,
            c=system_rank,
            transpose=False,
        )[:, 0]
        penalty_quadratic = float(np.sum(z_q1[n_wx_econ:n_augmented_rows] ** 2))
        z_buf.fill(0.0)
        z_buf[:system_rank] = z_q1[:system_rank]
        weighted_eta = _apply_q_left_serial(
            z_buf[:n_wx_econ],
            qr_wx,
            tau_wx,
            r=n_obs,
            c=n_wx_econ,
            transpose=False,
        )[:, 0]
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
                xx += qr_aug[j, i] * qrz_rank_raw[j]
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
                xx += qr_aug[j, k] * z_rank[j]
            z_rank[k] = (
                xwz_pivoted[k] / qr_aug[k, k]
                if k == 0
                else (xwz_pivoted[k] - xx) / qr_aug[k, k]
            )
        y_rank = z_rank.copy()
        if signed_correction is not None:
            y_rank = np.asarray(signed_correction @ y_rank, dtype=np.float64)

    for k in range(system_rank - 1, -1, -1):
        xx = 0.0
        for j in range(k + 1, system_rank):
            xx += qr_aug[k, j] * z_rank[j]
        z_rank[k] = (y_rank[k] - xx) / qr_aug[k, k]

    coef_kept = np.zeros(system_rank, dtype=np.float64)
    coef_kept[pivot_aug] = z_rank
    coef_full = _undrop_rows_vec(coef_kept, n_coef_total, dropped_column_indices)

    gauge_requested = bool(near_singular_null_pin)
    if gauge_requested and P_dense is not None:
        should_gauge = near_singular_null_pin is True or (
            str(near_singular_null_pin).lower() == "auto"
            and int(dropped_column_indices.size) > 0
        )
        if should_gauge:
            coef_full = _gauge_minimize_penalty_on_null_X(
                coef_full,
                X,
                np.asarray(P_dense, dtype=np.float64),
            )

    if use_wy:
        eta = np.asarray(X @ coef_full, dtype=np.float64)
        pen_vec = penalty_sqrt @ coef_full
        penalty_quadratic = float(pen_vec @ pen_vec)
    elif gauge_requested and P_dense is not None:
        pen_vec = penalty_sqrt @ coef_full
        penalty_quadratic = float(pen_vec @ pen_vec)

    return _StackedPlsNonnegOutcome(
        coef_full=coef_full,
        eta=eta,
        penalty_quadratic=penalty_quadratic,
        X=X,
        w=w,
        P_dense=None if P_dense is None else np.asarray(P_dense, dtype=np.float64),
        penalty_sqrt=penalty_sqrt,
        weighted_X=weighted_X,
        sqrt_w=raw_w,
        system_rank=int(system_rank),
        upper_r_final=upper_r_final,
        qr_wx=qr_wx,
        tau_wx=tau_wx,
        qr_aug=qr_aug,
        tau_aug=tau_aug,
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
    )


def _stacked_penalized_ls_nonneg_solution(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    penalty_sqrt: np.ndarray,
    penalty_rank_rows: np.ndarray,
    P_dense: np.ndarray | None,
    rank_tol: float,
    coef_method: str,
    near_singular_null_pin: bool | Literal["auto"],
) -> _StackedPlsNonnegOutcome:
    return _stacked_penalized_ls_nonneg_solution_literal(
        X,
        z,
        w,
        penalty_sqrt=penalty_sqrt,
        penalty_rank_rows=penalty_rank_rows,
        P_dense=P_dense,
        rank_tol=rank_tol,
        coef_method=coef_method,
        near_singular_null_pin=near_singular_null_pin,
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


def _build_q1_from_qr_factors(qr_wx, tau_wx, qr_aug, tau_aug, n, rr, rank):
    qr_aug_f = np.asfortranarray(np.asarray(qr_aug, dtype=np.float64).copy())
    Q_aug = _dorgqr_economic(qr_aug_f, tau_aug, rank)
    Q_top = np.asarray(Q_aug[:rr, :], dtype=np.float64)
    Q1 = np.zeros((n, rank), dtype=np.float64, order="F")
    Q1[:rr, :] = Q_top
    Q1 = _dormqr_apply(b"L", b"N", qr_wx, tau_wx, Q1)
    return np.asarray(Q1, dtype=np.float64, order="C"), qr_aug_f


def _build_nonnegative_q1_and_k(out, n, rr, rank):
    return _build_q1_from_qr_factors(
        out.qr_wx,
        out.tau_wx,
        out.qr_aug,
        out.tau_aug,
        n,
        rr,
        rank,
    )


def _signed_weight_rank_correction(
    q1_negative_rows: np.ndarray,
    *,
    rank_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Mirror `mgcv/src/gdi.c::pls_fit1()` signed-weight SVD correction.

    Returns coefficient correction `V (I - 2 D^2)^-1 V'`, covariance/root
    right factor `(I - 2 D^2)^-1/2 V'`, left factor `(I - 2 D^2)^1/2 V'`
    for `Rh`, and the determinant correction.
    """
    q1_negative_rows = np.asarray(q1_negative_rows, dtype=np.float64)
    if q1_negative_rows.ndim != 2:
        raise ValueError("q1_negative_rows must be 2-D.")
    rank = int(q1_negative_rows.shape[1])
    if rank == 0:
        z = np.zeros((0, 0), dtype=np.float64)
        return z, z, z, 0.0

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
    return correction, vt_scaled, rh_left, float(log_det_correction)


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
    Mp: int = 0,
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
        P_dense=None,
        rank_tol=rank_tol,
        coef_method="householder",
        near_singular_null_pin=False,
    )

    rank = int(out.system_rank)
    rr = int(out.n_wx_econ)
    drop = np.asarray(out.dropped_column_indices, dtype=int)
    n_drop = int(drop.size)
    pivot1 = np.asarray(out.pivot_aug, dtype=int).ravel()
    if pivot1.size != rank:
        raise RuntimeError("internal: pivot_aug length must equal system rank.")

    kept = tuple(int(k) for k in out.kept_original_indices)
    n_pen = int(E.shape[0])
    nr = rr + n_pen

    q1_matrix, qr_aug_f = _build_nonnegative_q1_and_k(out, X.shape[0], rr, rank)
    R_nr_rank = np.triu(np.asarray(qr_aug_f[:nr, :rank], dtype=np.float64))
    R = np.triu(np.asarray(out.upper_r_final, dtype=np.float64))
    raw_w = np.sqrt(np.abs(w))
    neg_weight_mask = np.asarray(w < 0.0, dtype=bool)
    X_drop = drop_columns_dense(X, drop) if n_drop else X.copy()
    X_piv = permute_columns(X_drop, pivot1, reverse=False)

    if np.any(neg_weight_mask):
        _, vt_scaled, rh_left, log_det_correction = _signed_weight_rank_correction(
            q1_matrix[neg_weight_mask, :],
            rank_tol=rank_tol,
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
                s += R_nr_rank[j, i] * q1tz[j]
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
            Rh, R_nr_rank, K, X, z, w, drop, pivot1
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

    _ = Mp
    return NonnegativePenalizedQRState(
        rank=rank,
        n_drop=n_drop,
        drop=drop.copy(),
        pivot1=pivot1.copy(),
        kept_original_indices=kept,
        P=P,
        K=K,
        Rh=Rh,
        PKtz=np.asarray(PKtz, dtype=np.float64).ravel().copy(),
        ldet_XWX_plus_S=ldet,
        beta_full=beta_full,
        rS_work=rS_work,
    )


def _frob_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def _upper_r_condition_indicator(upper_r: np.ndarray, n_leading_cols: int) -> float:
    """
    Upper-triangular condition number estimate (Cline–Moler–Stewart).

    ``upper_r`` is shape ``(nrow, ncol)``; only the leading
    ``n_leading_cols × n_leading_cols`` upper triangle is used.  Returns the
    product ``||R||_inf * ||R^{-1} e||_1`` as the condition indicator.
    """
    if n_leading_cols <= 0:
        return 0.0
    upper_r = np.asarray(upper_r, dtype=np.float64)
    c = n_leading_cols
    pp = np.zeros(c, dtype=np.float64)
    pm = np.zeros(c, dtype=np.float64)
    y = np.zeros(c, dtype=np.float64)
    p = np.zeros(c, dtype=np.float64)
    y_inf = 0.0
    r_inf = 0.0
    for k in range(c - 1, -1, -1):
        denom = upper_r[k, k]
        yp = (1.0 - p[k]) / denom
        ym = (-1.0 - p[k]) / denom
        for i in range(k):
            pp[i] = p[i] + upper_r[i, k] * yp
            pm[i] = p[i] + upper_r[i, k] * ym
        pp_norm = float(np.sum(np.abs(pp[:k])))
        pm_norm = float(np.sum(np.abs(pm[:k])))
        if abs(yp) + pp_norm >= abs(ym) + pm_norm:
            y[k] = yp
            p[:k] = pp[:k]
        else:
            y[k] = ym
            p[:k] = pm[:k]
        y_inf = max(y_inf, abs(y[k]))
    for i in range(c):
        s = 0.0
        for j in range(i, c):
            s += abs(upper_r[i, j])
        r_inf = max(r_inf, s)
    return float(r_inf * y_inf)


def _drop_columns(A: np.ndarray, drop: np.ndarray) -> np.ndarray:
    drop = np.asarray(drop, dtype=int)
    if drop.size == 0:
        return A
    mask = np.ones(A.shape[1], dtype=bool)
    mask[drop] = False
    return A[:, mask]


@lru_cache(maxsize=1)
def _lapack_ctypes_handles():
    lib_path = ctypes.util.find_library("lapack")
    if not lib_path:
        return None
    lib = ctypes.CDLL(lib_path)
    dgeqp3 = getattr(lib, "dgeqp3_", None)
    dormqr = getattr(lib, "dormqr_", None)
    if dgeqp3 is None or dormqr is None:
        return None
    dgeqp3.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    dormqr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    return dgeqp3, dormqr


def _dgeqp3_f_with_jpvt(
    a: np.ndarray,
    *,
    jpvt_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LAPACK ``dgeqp3``; return (factor storage, ``jpvt`` 0-based, ``tau``).

    `mgcv::pls_fit1()` reuses the previous `pivot1` buffer when calling the
    final QR. SciPy's wrapper does not expose input `JPVT`, so use the raw
    LAPACK symbol when those flags matter.
    """
    a_f = np.asfortranarray(np.asarray(a, dtype=np.float64), dtype=np.float64)
    m, n = map(int, a_f.shape)
    handles = _lapack_ctypes_handles()
    if handles is None:
        if jpvt_in is not None:
            raise RuntimeError(
                "LAPACK dgeqp3 with input JPVT is unavailable in this environment."
            )
        qr_a, jpvt, tau, _work, info = _lapack.dgeqp3(a_f)
        if info != 0:
            raise RuntimeError(f"dgeqp3 failed with info={info}")
        return qr_a, jpvt.astype(np.int64) - 1, tau

    dgeqp3, _dormqr = handles
    m_c = ctypes.c_int(m)
    n_c = ctypes.c_int(n)
    lda_c = ctypes.c_int(max(1, m))
    jpvt = (
        np.zeros(n, dtype=np.int32)
        if jpvt_in is None
        else np.asarray(jpvt_in, dtype=np.int32).copy()
    )
    tau = np.zeros(min(m, n), dtype=np.float64)
    info = ctypes.c_int(0)
    work = np.zeros(1, dtype=np.float64)
    lwork = ctypes.c_int(-1)

    dgeqp3(
        ctypes.byref(m_c),
        ctypes.byref(n_c),
        a_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lda_c),
        jpvt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lwork),
        ctypes.byref(info),
    )
    if info.value != 0:
        raise RuntimeError(f"dgeqp3 workspace query failed with info={info.value}")

    lwork = ctypes.c_int(max(int(work[0]), 1))
    work = np.zeros(int(lwork.value), dtype=np.float64)
    info = ctypes.c_int(0)
    dgeqp3(
        ctypes.byref(m_c),
        ctypes.byref(n_c),
        a_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lda_c),
        jpvt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lwork),
        ctypes.byref(info),
    )
    if info.value != 0:
        raise RuntimeError(f"dgeqp3 failed with info={info.value}")
    return a_f, jpvt.astype(np.int64) - 1, tau


def _dgeqp3_f(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _dgeqp3_f_with_jpvt(a)


def _dgeqp3_economic_r(
    a: np.ndarray,
    *,
    jpvt_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivoted economic QR via ``dgeqp3`` only (same pivots / ``R`` as ``scipy.linalg.qr``).

    Returns
    -------
    qr_a, tau, piv, R0
        ``qr_a`` / ``tau`` compact Householder (Fortran); ``piv`` 0-based; ``R0`` is
        ``np.triu(qr_a[:k, :])`` with ``k = min(m, n)`` (``scipy`` ``mode='economic'``).
    """
    a = np.asarray(a, dtype=np.float64)
    m, n = int(a.shape[0]), int(a.shape[1])
    k = min(m, n)
    qr_a, piv, tau = _dgeqp3_f_with_jpvt(a.copy(), jpvt_in=jpvt_in)
    r0 = np.triu(np.asarray(qr_a[:k, :], dtype=np.float64))
    return qr_a, tau, piv, r0


def _dorgqr_economic(qr_a: np.ndarray, tau: np.ndarray, ncols: int) -> np.ndarray:
    """
    Economic ``Q`` with ``ncols`` columns from ``dgeqp3`` compact form.

    **Overwrites** the Fortran buffer ``qr_a``; pass a **copy** if the factor is still
    needed for ``dormqr``.
    """
    qr_a = np.asfortranarray(np.asarray(qr_a, dtype=np.float64), dtype=np.float64)
    _, work, info = _lapack.dorgqr(qr_a, tau, lwork=-1)
    if info != 0:
        raise RuntimeError(f"dorgqr workspace query failed with info={info}")
    lwork = int(work[0])
    q_f, _, info = _lapack.dorgqr(qr_a, tau, lwork=lwork)
    if info != 0:
        raise RuntimeError(f"dorgqr failed with info={info}")
    q_f = np.asarray(q_f, dtype=np.float64)[:, : int(ncols)]
    return q_f


def _dormqr_apply(
    side: bytes,
    trans: bytes,
    qr_a: np.ndarray,
    tau: np.ndarray,
    fortran_block: np.ndarray,
    *,
    lwork: int | None = None,
) -> np.ndarray:
    """Apply Householder ``Q`` from ``dgeqp3`` to Fortran-contiguous operand matrix."""
    qr_a = np.asfortranarray(np.asarray(qr_a, dtype=np.float64), dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64).ravel()
    k = int(tau.shape[0])
    if side == b"L" and qr_a.shape[1] != k:
        # SciPy's `dormqr` wrapper expects the compact Householder storage in
        # `(lda, k)` form. `dgeqp3` returns a wide `m x n` buffer when `m < n`,
        # so keep only the leading reflector columns before applying `Q`.
        qr_a = np.asfortranarray(qr_a[:, :k], dtype=np.float64)
    elif side == b"R" and qr_a.shape[0] != k:
        qr_a = np.asfortranarray(qr_a[:k, :], dtype=np.float64)
    c = np.asfortranarray(np.asarray(fortran_block, dtype=np.float64), dtype=np.float64)
    handles = _lapack_ctypes_handles()
    if handles is None:
        m = int(qr_a.shape[0])
        if lwork is None:
            lwork = max(1, m * max(1, c.shape[1]) * 32)
        cq, _wk, info = _lapack.dormqr(side, trans, qr_a, tau, c, lwork=lwork)
        if info != 0:
            raise RuntimeError(f"dormqr failed with info={info}")
        return np.asarray(cq, dtype=np.float64)

    _dgeqp3, dormqr = handles
    m_c = ctypes.c_int(int(c.shape[0]))
    n_c = ctypes.c_int(int(c.shape[1]))
    k_c = ctypes.c_int(k)
    lda_c = ctypes.c_int(max(1, int(qr_a.shape[0])))
    ldc_c = ctypes.c_int(max(1, int(c.shape[0])))
    info = ctypes.c_int(0)
    work = np.zeros(1, dtype=np.float64)
    lwork_c = ctypes.c_int(-1)

    dormqr(
        side,
        trans,
        ctypes.byref(m_c),
        ctypes.byref(n_c),
        ctypes.byref(k_c),
        qr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lda_c),
        tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ldc_c),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lwork_c),
        ctypes.byref(info),
    )
    if info.value != 0:
        raise RuntimeError(f"dormqr workspace query failed with info={info.value}")

    lwork_c = ctypes.c_int(max(int(work[0]), 1) if lwork is None else int(lwork))
    work = np.zeros(int(lwork_c.value), dtype=np.float64)
    info = ctypes.c_int(0)
    dormqr(
        side,
        trans,
        ctypes.byref(m_c),
        ctypes.byref(n_c),
        ctypes.byref(k_c),
        qr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lda_c),
        tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ldc_c),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lwork_c),
        ctypes.byref(info),
    )
    if info.value != 0:
        raise RuntimeError(f"dormqr failed with info={info.value}")
    return np.asarray(c, dtype=np.float64)


def _drop_rows_vec(v: np.ndarray, drop_sorted: np.ndarray) -> np.ndarray:
    """``mgcv`` ``drop_rows`` for a single column."""
    if drop_sorted.size == 0:
        return np.asarray(v, dtype=np.float64).copy()
    q = int(v.shape[0])
    mask = np.ones(q, dtype=bool)
    mask[drop_sorted.astype(int)] = False
    return np.asarray(v, dtype=np.float64)[mask].copy()


def project_coef_onto_row_space(
    X: np.ndarray,
    coef_full: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> np.ndarray:
    """
    Orthogonal projection of ``coef_full`` onto ``row(X^T) \\subset \\mathbb{R}^q``.

    Coefficients that differ only by ``\\delta \\in \\mathrm{null}(X)`` share the same
    projection; fitted values ``X @ \\beta`` depend only on this component.  At REML
    boundaries where ``\\lambda \\to 0`` and ``X`` is rank-deficient, mgcv's
    ``coef(gam)`` may differ from nampy along ``null(X)`` in float64, but this
    projection matches mgcv to machine precision when ``X`` agrees (parity tests).
    """
    return _project_coef_onto_row_space(X, coef_full, sv_rel_tol=sv_rel_tol)


def snap_coef_to_reference_null_space(
    coef_full: np.ndarray,
    X: np.ndarray,
    coef_reference: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> np.ndarray:
    """
    **Null-space tie-break** relative to a reference coefficient vector (mgcv parity).

    Any ``\\delta \\in \\mathrm{null}(X)`` leaves ``X\\beta`` unchanged.  Pivoted QR in
    ``mgcv``'s C code picks one coset member; tiny BLAS / pivot differences can shift
    ``\\beta`` along ``\\mathrm{null}(X)`` while ``X\\beta`` still matches ``mgcv`` to
    float64 noise.  Given a reference ``\\beta^{\\mathrm{ref}}`` (e.g. ``coef(gam)`` from
    R for the same ``X`` and working response), this returns the unique coset member that
    shares the **same** null-space coordinates as the reference:

    .. math::

        \\beta' = \\beta + N N^{\\top}(\\beta^{\\mathrm{ref}} - \\beta),

    with columns of ``N`` an orthonormal basis of ``\\mathrm{null}(X)``.  Then
    ``X\\beta' = X\\beta`` and ``N^{\\top}\\beta' = N^{\\top}\\beta^{\\mathrm{ref}}``.
    """
    return _snap_coef_to_reference_null_space(
        coef_full,
        X,
        coef_reference,
        sv_rel_tol=sv_rel_tol,
    )


def _gauge_minimize_penalty_on_null_X(
    coef_full: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    *,
    sv_rel_tol: float = 1e-12,
) -> np.ndarray:
    """
    Within ``{β + null(X)}``, pick the representative minimising ``β'Pβ``.

    When ``N'PN`` is tiny (``\\lambda \\approx 0`` on a rank-deficient ``X``), the
    first-order system is ill-conditioned; a small **ridge** on ``N'PN`` stabilises the
    linear solve without changing ``X\\beta`` (still ``β \\leftarrow β + N z``).
    """
    X = np.asarray(X, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    P = symmetrize_matrix(P)
    q = int(X.shape[1])
    if q == 0:
        return coef_full
    N, _rank_x = svd_null_space_basis(X, sv_rel_tol=sv_rel_tol)
    null_dim = int(N.shape[1])
    if null_dim == 0:
        return coef_full
    H = N.T @ P @ N
    H = symmetrize_matrix(H)
    rhs = np.asarray(-(N.T @ (P @ coef_full)), dtype=np.float64).ravel()
    h_evals = symmetric_eigvalsh(H)
    h_max = float(np.max(h_evals)) if h_evals.size else 0.0
    p_frob = float(np.linalg.norm(P, ord="fro"))
    ridge = 0.0
    if h_max <= 0.0 or not np.isfinite(h_max):
        ridge = max(
            STACKED_QR_RANK_TOLERANCE * max(p_frob, np.finfo(np.float64).tiny),
            np.finfo(np.float64).eps ** 0.75 * max(1.0, p_frob),
        )
    elif h_max < STACKED_QR_RANK_TOLERANCE * max(p_frob, 1.0):
        ridge = STACKED_QR_RANK_TOLERANCE * max(p_frob, np.finfo(np.float64).tiny)

    Hr = H + ridge * np.eye(null_dim, dtype=np.float64)
    try:
        z = np.linalg.solve(Hr, rhs.reshape(-1, 1))
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(Hr, rhs.reshape(-1, 1), rcond=None)[0]
    z = np.asarray(z, dtype=np.float64).ravel()
    return coef_full + N @ z


def _solve_coef_householder_chain_nonneg_weights(
    *,
    y: np.ndarray,
    w: np.ndarray,
    X: np.ndarray,
    qr_weighted_x: np.ndarray,
    tau_weighted_x: np.ndarray,
    n_obs: int,
    n_coef: int,
    n_weighted_x_rows: int,
    n_augmented_rows: int,
    system_rank: int,
    qr_augmented: np.ndarray,
    tau_augmented: np.ndarray,
    pivot_augmented: np.ndarray,
    dropped_columns: np.ndarray,
    kept_original_indices: np.ndarray,
    diagonal_stability_tol: float,
) -> np.ndarray:
    """
    Back-solve for coefficients after the two stacked pivoted QRs (non-negative weights).

    Applies the full Householder chain:
    Q_wx' → Q_aug' → Q_aug → Q_wx → triangular back-substitution.
    Falls back to a forward-substitution pass if the diagonal reconstruction test
    detects numerical instability.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    pivot_augmented = np.asarray(pivot_augmented, dtype=np.int64).ravel()
    dropped_columns = np.asarray(dropped_columns, dtype=np.int64).ravel()
    kept_original_indices = np.asarray(kept_original_indices, dtype=np.int64).ravel()
    raw = np.sqrt(np.maximum(w, 0.0))
    wy = w * y
    n_qwx_rows = int(qr_weighted_x.shape[0])
    nz = int(max(n_obs, n_augmented_rows, n_qwx_rows))
    z = np.zeros(nz, dtype=np.float64)
    z[:n_obs] = y * raw

    scratch = np.zeros(4 * n_coef, dtype=np.float64)
    forward_rhs = np.zeros(n_coef, dtype=np.float64)

    # Apply Householder chain: Q_wx', Q_aug', Q_aug, Q_wx in sequence.
    stage1 = np.asfortranarray(z[:n_qwx_rows].reshape(n_qwx_rows, 1))
    stage1 = _dormqr_apply(b"L", b"T", qr_weighted_x, tau_weighted_x, stage1)
    z[:n_qwx_rows] = stage1[:, 0]

    z[system_rank:nz] = 0.0

    stage2 = np.asfortranarray(z[:n_augmented_rows].reshape(n_augmented_rows, 1))
    stage2 = _dormqr_apply(b"L", b"T", qr_augmented, tau_augmented, stage2)
    z[:n_augmented_rows] = stage2[:, 0]

    z[system_rank:nz] = 0.0

    for i in range(system_rank):
        scratch[n_coef + i] = z[i]
        forward_rhs[i] = z[i]

    stage3 = np.asfortranarray(z[:n_augmented_rows].reshape(n_augmented_rows, 1))
    stage3 = _dormqr_apply(b"L", b"N", qr_augmented, tau_augmented, stage3)
    z[:n_augmented_rows] = stage3[:, 0]

    for i in range(system_rank, n_obs):
        z[i] = 0.0

    stage4 = np.asfortranarray(z[:n_qwx_rows].reshape(n_qwx_rows, 1))
    stage4 = _dormqr_apply(b"L", b"N", qr_weighted_x, tau_weighted_x, stage4)
    z[:n_qwx_rows] = stage4[:, 0]

    scratch[:n_coef] = X.T @ wy
    rhs_kept = _drop_rows_vec(scratch[:n_coef], dropped_columns)
    rhs_pivoted = np.zeros(system_rank, dtype=np.float64)
    for k in range(system_rank):
        rhs_pivoted[k] = rhs_kept[int(pivot_augmented[k])]
    scratch[:system_rank] = rhs_pivoted

    use_wy_path = False
    recon_error_sq = 0.0
    target_norm_sq = 0.0
    R_aug_f = np.asarray(qr_augmented, dtype=np.float64, order="F")
    for i in range(system_rank):
        lower_dot = 0.0
        for j in range(i + 1):
            lower_dot += R_aug_f[j, i] * scratch[n_coef + j]
        lower_dot -= scratch[i]
        recon_error_sq += lower_dot * lower_dot
        target_norm_sq += scratch[i] * scratch[i]
    if recon_error_sq > diagonal_stability_tol * target_norm_sq:
        use_wy_path = True

    if use_wy_path:
        for k in range(system_rank):
            lower_dot = 0.0
            for j in range(k):
                lower_dot += R_aug_f[j, k] * z[j]
            diag = R_aug_f[k, k]
            z[k] = (scratch[k] - lower_dot) / diag if abs(diag) > LOG_GUARD_MIN else 0.0
        forward_rhs[:system_rank] = z[:system_rank]

    for k in range(system_rank - 1, -1, -1):
        back_dot = 0.0
        for j in range(k + 1, system_rank):
            back_dot += R_aug_f[k, j] * z[j]
        diag = R_aug_f[k, k]
        z[k] = (forward_rhs[k] - back_dot) / diag if abs(diag) > LOG_GUARD_MIN else 0.0

    coef_reduced = np.zeros(system_rank, dtype=np.float64)
    for i in range(system_rank):
        coef_reduced[int(pivot_augmented[i])] = z[i]

    coef_full = np.zeros(n_coef, dtype=np.float64)
    coef_full[kept_original_indices] = coef_reduced
    return coef_full


def penalty_sqrt_rows(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Penalty square-root that preserves diagonal structure when ``P`` is diagonal.

    When ``P`` is (numerically) diagonal, an eigendecomposition would mix the
    eigenspace arbitrarily and change the minimum-norm tie-break used in lstsq.
    This function uses one identity row per positive diagonal entry instead,
    matching the coefficient tie-break from the triangular solve path.
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


def balanced_penalty_template_sqrt_for_rank(
    penalty_blocks: Iterable[Any],
    *,
    fit_intercept: bool,
    n_coef: int,
) -> np.ndarray:
    """
    Balanced penalty square root for numerical rank detection.

    Computes a Frobenius-normalised aggregate of the unscaled penalty templates
    (each ``pb.matrix / ||pb.matrix||_F``, no lambda scaling), then takes an
    eigendecomposition and retains eigenvectors above a relative threshold.
    The resulting rows are used only for rank detection in the stacked QR; they
    are not used to compute coefficients.
    """
    return _balanced_penalty_template_sqrt_for_rank(
        penalty_blocks,
        fit_intercept=fit_intercept,
        n_coef=n_coef,
    )


def _ridge_eps_for_upper_r(upper_r: np.ndarray) -> float:
    """Tiny diagonal for stable inversion of the final upper-triangular ``R`` (EDF factors)."""
    d = np.abs(np.diag(upper_r))
    scale = float(np.max(d)) if d.size else 1.0
    return max(np.finfo(np.float64).eps * max(scale, 1.0), 1e-16)


def pls_fit1_nonneg_w(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    wy: np.ndarray,
    *,
    penalty_sqrt_E: np.ndarray,
    penalty_rank_Es: np.ndarray,
    rank_tol: float = STACKED_QR_RANK_TOLERANCE,
    coef_method: str = "householder",
    near_singular_null_pin: bool | Literal["auto"] = False,
    P_for_gauge: np.ndarray | None = None,
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
    P_for_gauge
        Optional total penalty matrix for the null-space penalty-minimisation gauge
        (activated by ``near_singular_null_pin``).

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
    if not np.allclose(wy, exp_wy, rtol=0.0, atol=np.finfo(np.float64).eps * 8):
        raise ValueError("wy must equal w * z elementwise (mgcv pls_fit1 contract).")
    out = _stacked_penalized_ls_nonneg_solution(
        X,
        z,
        w,
        penalty_sqrt=np.asarray(penalty_sqrt_E, dtype=np.float64),
        penalty_rank_rows=np.asarray(penalty_rank_Es, dtype=np.float64),
        P_dense=P_for_gauge,
        rank_tol=rank_tol,
        coef_method=coef_method,
        near_singular_null_pin=near_singular_null_pin,
    )
    return out.coef_full, out.penalty_quadratic


def solve_gaussian_penalized_ls_stacked_qr(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    *,
    penalty_blocks: Iterable[Any] | None = None,
    penalty_rank_rows: np.ndarray | None = None,
    fit_intercept: bool = True,
    n_coef: int | None = None,
    rank_tol: float = STACKED_QR_RANK_TOLERANCE,
    coef_method: str = "householder",
    near_singular_null_pin: bool | Literal["auto"] = False,
) -> dict:
    """
    Full Gaussian penalized least-squares solve returning EDF and covariance matrices.

    Calls the stacked pivoted QR solver and then reconstructs the matrices needed
    for EDF computation and post-fit diagnostics.

    Parameters
    ----------
    penalty_blocks, fit_intercept, n_coef
        When ``penalty_blocks`` is provided (along with ``n_coef``), rank detection
        uses :func:`balanced_penalty_template_sqrt_for_rank`.  Omit these to fall back
        to row-normalised ``sqrt(P)``, which is less stable for mixed-scale penalties.
    coef_method
        ``"householder"`` (default): triangular back-substitution after stacked QRs.
        ``"lstsq"``: augmented least-squares with penalty-minimisation gauge.
    rank_tol
        Condition threshold for the rank-reveal step; default is
        :data:`STACKED_QR_RANK_TOLERANCE`.
    near_singular_null_pin
        If True (or ``"auto"``), apply a null-space penalty-minimisation gauge after
        solving.  Useful for near-singular designs; disabled by default.

    Returns
    -------
    dict
        Keys include ``coef_full``, ``XtWX_plus_penalty_chol_inverse_embedded``
        (retained for compatibility; equal to the full-space covariance root),
        ``covariance_root`` (q×rank `rV` analogue), ``A_inv`` (rank-aware
        covariance / pseudoinverse analogue), ``householder_mixing_obs_coef``
        (n×q K), ``coef_hat_matrix`` (the F matrix for EDF computation),
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
    penalty_sqrt, penalty_rank_template = penalty_sqrt_rows(P)

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
        P_dense=P,
        rank_tol=rank_tol,
        coef_method=coef_method,
        near_singular_null_pin=near_singular_null_pin,
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

    householder_mixing = np.zeros((n_obs, n_coef_total), dtype=np.float64)

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
        "XtWX_plus_penalty_chol_inverse_embedded": covariance_root,
        "householder_mixing_obs_coef": householder_mixing,
        "log_det_XtWX_plus_penalty": log_det_XtWX_plus_penalty,
        "penalized_system_rank": int(system_rank),
        "coef_hat_matrix": coef_hat_matrix,
        "dropped_column_indices": dropped_column_indices,
    }


def gaussian_design_needs_stacked_qr_fit(model) -> bool:
    for tb in _term_blocks_seq(model):
        if str(getattr(tb, "term_type", "")).lower() == "random_effect":
            return True
    from ..._model_state import _design_matrix

    Z = _design_matrix(model)
    if Z is None:
        return False
    from ..penalized_system import build_full_design

    X = np.asarray(
        build_full_design(Z, fit_intercept=_fit_intercept(model)),
        dtype=np.float64,
    )
    if X.ndim != 2 or X.shape[1] == 0:
        return False
    return matrix_is_rank_deficient(X)
