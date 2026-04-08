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
R factor.  The threshold is :data:`STACKED_QR_RANK_TOLERANCE` (100 * machine eps).
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

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import lapack as _lapack

# Rank-detection threshold: condition number ratio above which a column is considered
# linearly dependent.  100 * machine eps is the standard choice for this algorithm.
STACKED_QR_RANK_TOLERANCE = np.finfo(np.float64).eps * 100.0

# Below this Frobenius norm, the penalty is numerically zero on most of the null space
# of X (e.g. a random-effect term at the lower smoothing parameter bound).  The
# stacked-QR path still returns a valid coset member, but a penalty-minimisation
# gauge on null(X) improves stability when ``near_singular_null_pin`` is enabled.
NEAR_SINGULAR_PENALTY_FROB_TOL = 1e-20


@dataclass(frozen=True)
class _StackedPlsNonnegOutcome:
    """Internal: one non-negative-weight stacked PLS solve + QR state for post-processing."""

    coef_full: np.ndarray
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
    """
    Core stacked pivoted QR + Householder coefficient solve (non-negative weights).

    Used by :func:`pls_fit1_nonneg_w` and :func:`solve_gaussian_penalized_ls_stacked_qr`.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n_obs, n_coef_total = X.shape
    if z.shape[0] != n_obs or w.shape[0] != n_obs:
        raise ValueError("shape mismatch between X rows, z, and w.")
    if np.any(w < 0):
        raise ValueError("stacked PLS requires non-negative weights.")
    penalty_sqrt = np.asarray(penalty_sqrt, dtype=np.float64)
    penalty_rank_rows = np.asarray(penalty_rank_rows, dtype=np.float64)
    n_penalty_rows = int(penalty_sqrt.shape[0])
    n_rank_penalty_rows = int(penalty_rank_rows.shape[0])
    if n_penalty_rows == 0 or n_rank_penalty_rows == 0:
        raise ValueError("penalty sqrt matrices must be non-empty.")

    sqrt_w = np.sqrt(w)
    weighted_X = sqrt_w[:, None] * X
    n_wx_econ = min(n_obs, n_coef_total)

    qr_wx, tau_wx, pivot_wx, r_wx_econ = _dgeqp3_economic_r(weighted_X)
    R_weighted_natural = np.zeros((n_wx_econ, n_coef_total), dtype=np.float64)
    for j in range(n_coef_total):
        R_weighted_natural[:, int(pivot_wx[j])] = r_wx_econ[:, j]

    frob_Rw = _frob_norm(R_weighted_natural)
    frob_Es = _frob_norm(penalty_rank_rows)
    if frob_Rw <= 0.0:
        frob_Rw = 1.0
    if frob_Es <= 0.0:
        frob_Es = 1.0

    n_stack_rows = n_wx_econ + n_rank_penalty_rows
    rank_stack = np.zeros((n_stack_rows, n_coef_total), dtype=np.float64)
    rank_stack[:n_wx_econ, :] = R_weighted_natural / frob_Rw
    rank_stack[n_wx_econ:, :] = penalty_rank_rows / frob_Es

    _, _, pivot_rank_reveal, r_rank_econ = _dgeqp3_economic_r(rank_stack)

    system_rank = min(n_coef_total, n_stack_rows)
    rcond = _upper_r_condition_indicator(r_rank_econ, system_rank)
    while system_rank > 0 and rank_tol * rcond > 1.0:
        system_rank -= 1
        rcond = _upper_r_condition_indicator(r_rank_econ, system_rank)

    if n_coef_total > system_rank:
        dropped_column_indices = np.sort(
            np.asarray(
                [int(pivot_rank_reveal[i]) for i in range(system_rank, n_coef_total)],
                dtype=int,
            )
        )
    else:
        dropped_column_indices = np.zeros((0,), dtype=int)

    R_weighted_kept = _drop_columns(R_weighted_natural, dropped_column_indices)
    penalty_sqrt_kept = _drop_columns(penalty_sqrt, dropped_column_indices)
    drop_set = set(int(d) for d in dropped_column_indices)
    kept_original_indices = [k for k in range(n_coef_total) if k not in drop_set]

    n_augmented_rows = n_wx_econ + n_penalty_rows
    augmented_r_block = np.zeros((n_augmented_rows, system_rank), dtype=np.float64)
    augmented_r_block[:n_wx_econ, :] = R_weighted_kept
    augmented_r_block[n_wx_econ:, :] = penalty_sqrt_kept

    qr_aug, tau_aug, pivot_aug, r_aug_econ = _dgeqp3_economic_r(augmented_r_block)
    upper_r_final = np.triu(
        np.asarray(r_aug_econ[:system_rank, :system_rank], dtype=np.float64)
    )

    cm = str(coef_method).lower().strip()
    if cm == "householder":
        kept_arr = np.asarray(kept_original_indices, dtype=np.int64)
        coef_full = _solve_coef_householder_chain_nonneg_weights(
            y=z,
            w=w,
            X=X,
            qr_weighted_x=qr_wx,
            tau_weighted_x=tau_wx,
            n_obs=n_obs,
            n_coef=n_coef_total,
            n_weighted_x_rows=n_wx_econ,
            n_augmented_rows=n_augmented_rows,
            system_rank=system_rank,
            qr_augmented=qr_aug,
            tau_augmented=tau_aug,
            pivot_augmented=pivot_aug,
            dropped_columns=dropped_column_indices,
            kept_original_indices=kept_arr,
            diagonal_stability_tol=STACKED_QR_RANK_TOLERANCE,
        )
        pin = near_singular_null_pin
        if P_dense is not None:
            p_frob = float(np.linalg.norm(P_dense, ord="fro"))
            if pin is True or (
                pin == "auto" and p_frob < NEAR_SINGULAR_PENALTY_FROB_TOL
            ):
                coef_full = _gauge_minimize_penalty_on_null_X(coef_full, X, P_dense)
    elif cm == "lstsq":
        design_kept = _drop_columns(X, dropped_column_indices)
        aug_design = np.vstack([sqrt_w[:, None] * design_kept, penalty_sqrt_kept])
        rhs_vec = np.concatenate(
            [sqrt_w * z, np.zeros(penalty_sqrt_kept.shape[0], dtype=np.float64)]
        )
        coef_sub, *_ = np.linalg.lstsq(aug_design, rhs_vec, rcond=None)
        coef_full = np.zeros(n_coef_total, dtype=np.float64)
        for j, kcol in enumerate(kept_original_indices):
            coef_full[kcol] = coef_sub[j]
        if P_dense is not None:
            coef_full = _gauge_minimize_penalty_on_null_X(coef_full, X, P_dense)
    else:
        raise ValueError(
            f"Unknown coef_method {coef_method!r}; use 'householder' or 'lstsq'."
        )

    pen_vec = penalty_sqrt @ coef_full
    penalty_quadratic = float(pen_vec @ pen_vec)

    return _StackedPlsNonnegOutcome(
        coef_full=coef_full,
        penalty_quadratic=penalty_quadratic,
        X=X,
        w=w,
        P_dense=P_dense,
        penalty_sqrt=penalty_sqrt,
        weighted_X=weighted_X,
        sqrt_w=sqrt_w,
        system_rank=int(system_rank),
        upper_r_final=upper_r_final,
        qr_wx=qr_wx,
        tau_wx=tau_wx,
        qr_aug=qr_aug,
        tau_aug=tau_aug,
        pivot_aug=pivot_aug.astype(np.int64),
        n_wx_econ=int(n_wx_econ),
        kept_original_indices=kept_original_indices,
        dropped_column_indices=dropped_column_indices,
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


def _dgeqp3_f(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAPACK ``dgeqp3``; return (factor storage, ``jpvt`` 0-based, ``tau``)."""
    a = np.asfortranarray(np.asarray(a, dtype=np.float64), dtype=np.float64)
    qr_a, jpvt, tau, _work, info = _lapack.dgeqp3(a)
    if info != 0:
        raise RuntimeError(f"dgeqp3 failed with info={info}")
    return qr_a, jpvt.astype(np.int64) - 1, tau


def _dgeqp3_economic_r(
    a: np.ndarray,
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
    qr_a, piv, tau = _dgeqp3_f(a.copy())
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
    c = np.asfortranarray(np.asarray(fortran_block, dtype=np.float64), dtype=np.float64)
    m = int(qr_a.shape[0])
    if lwork is None:
        lwork = max(1, m * max(1, c.shape[1]) * 32)
    cq, _wk, info = _lapack.dormqr(side, trans, qr_a, tau, c, lwork=lwork)
    if info != 0:
        raise RuntimeError(f"dormqr failed with info={info}")
    return np.asarray(cq, dtype=np.float64)


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
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(coef_full, dtype=np.float64).ravel()
    q = int(X.shape[1])
    if b.shape[0] != q:
        raise ValueError(f"coef_full has length {b.shape[0]}, expected {q}.")
    if q == 0:
        return b.copy()
    _, s, Vt = np.linalg.svd(X, full_matrices=True)
    if s.size == 0:
        return b.copy()
    smax = float(s[0])
    tol = max(smax * sv_rel_tol, np.finfo(np.float64).eps * max(smax, 1.0))
    rank_x = int(np.sum(s > tol))
    if rank_x >= q:
        return b.copy()
    N = Vt[rank_x:q, :].T
    return b - N @ (N.T @ b)


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
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(coef_full, dtype=np.float64).ravel()
    ref = np.asarray(coef_reference, dtype=np.float64).ravel()
    _, q = X.shape
    if b.shape[0] != q or ref.shape[0] != q:
        raise ValueError("coef vectors must have length q matching X.shape[1].")
    if q == 0:
        return b.copy()
    _, s, Vt = np.linalg.svd(X, full_matrices=True)
    if s.size == 0:
        return b.copy()
    smax = float(s[0])
    tol = max(smax * sv_rel_tol, np.finfo(np.float64).eps * max(smax, 1.0))
    rank_x = int(np.sum(s > tol))
    if rank_x >= q:
        return b.copy()
    N = np.asarray(Vt[rank_x:q, :], dtype=np.float64).T
    pn = N @ N.T
    return b + pn @ (ref - b)


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
    P = 0.5 * (P + P.T)
    q = int(X.shape[1])
    if q == 0:
        return coef_full
    _, s, Vt = np.linalg.svd(X, full_matrices=True)
    if s.size == 0:
        return coef_full
    smax = float(s[0])
    tol = max(smax * sv_rel_tol, np.finfo(np.float64).eps * max(smax, 1.0))
    rank_x = int(np.sum(s > tol))
    null_dim = q - rank_x
    if null_dim == 0:
        return coef_full
    N = Vt[rank_x:q, :].T
    H = N.T @ P @ N
    H = 0.5 * (H + H.T)
    rhs = np.asarray(-(N.T @ (P @ coef_full)), dtype=np.float64).ravel()
    h_evals = np.linalg.eigvalsh(H)
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
            z[k] = (scratch[k] - lower_dot) / diag if abs(diag) > 1e-300 else 0.0
        forward_rhs[:system_rank] = z[:system_rank]

    for k in range(system_rank - 1, -1, -1):
        back_dot = 0.0
        for j in range(k + 1, system_rank):
            back_dot += R_aug_f[k, j] * z[j]
        diag = R_aug_f[k, k]
        z[k] = (forward_rhs[k] - back_dot) / diag if abs(diag) > 1e-300 else 0.0

    coef_reduced = np.zeros(system_rank, dtype=np.float64)
    for i in range(system_rank):
        coef_reduced[int(pivot_augmented[i])] = z[i]

    coef_full = np.zeros(n_coef, dtype=np.float64)
    coef_full[kept_original_indices] = coef_reduced
    return coef_full


def penalty_sqrt_rows(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Penalty square-root factor ``E`` with ``E.T @ E = P`` (``P`` symmetric PSD),
    plus a row-normalised version ``Es`` used for rank detection in the stacked QR.
    """
    P = np.asarray(P, dtype=np.float64)
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    mask = w > max(np.max(w) * 1e-15, 1e-300)
    if not np.any(mask):
        return np.zeros((0, P.shape[0])), np.zeros((0, P.shape[0]))
    wl = np.sqrt(np.maximum(w[mask], 0.0))
    Vp = V[:, mask]
    E = (wl[:, None] * Vp.T)  # r × q
    row_norms = np.linalg.norm(E, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-300)
    Es = E / row_norms
    return E, Es


def penalty_sqrt_rows_prefer_diagonal(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Penalty square-root that preserves diagonal structure when ``P`` is diagonal.

    When ``P`` is (numerically) diagonal, an eigendecomposition would mix the
    eigenspace arbitrarily and change the minimum-norm tie-break used in lstsq.
    This function uses one identity row per positive diagonal entry instead,
    matching the coefficient tie-break from the triangular solve path.
    Falls back to :func:`penalty_sqrt_rows` when ``P`` has off-diagonal entries.
    """
    P = np.asarray(P, dtype=np.float64)
    P = 0.5 * (P + P.T)
    q = int(P.shape[0])
    d = np.diag(P)
    off = P - np.diag(d)
    scale = float(np.max(np.abs(P))) if q else 1.0
    if scale <= 0.0:
        scale = 1.0
    if not np.allclose(off, 0.0, atol=1e-14 * max(scale, 1.0), rtol=0.0):
        return penalty_sqrt_rows(P)
    rows: list[np.ndarray] = []
    thr = max(np.max(d) * 1e-15, 1e-300)
    for j in range(q):
        if d[j] > thr:
            r = np.zeros(q, dtype=np.float64)
            r[j] = float(np.sqrt(max(d[j], 0.0)))
            rows.append(r)
    if not rows:
        return np.zeros((0, q)), np.zeros((0, q))
    E = np.vstack(rows)
    row_norms = np.linalg.norm(E, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-300)
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
    offset = 1 if fit_intercept else 0
    q = offset + int(n_coef)
    template_sum = np.zeros((q, q), dtype=np.float64)
    for pb in penalty_blocks:
        smooth_idx = int(getattr(pb, "smoothing_index", -1))
        if smooth_idx < 0:
            continue
        template_block = np.asarray(pb.matrix, dtype=np.float64)
        sl = pb.coef_slice
        block_dim = int(sl.stop - sl.start)
        if template_block.shape != (block_dim, block_dim):
            continue
        frob = float(np.linalg.norm(template_block, ord="fro"))
        if frob <= 0.0:
            continue
        idx = np.arange(offset + sl.start, offset + sl.stop, dtype=np.int64)
        template_sum[np.ix_(idx, idx)] += template_block / frob
    template_sum = 0.5 * (template_sum + template_sum.T)
    evals, evecs = np.linalg.eigh(template_sum)
    emax = float(np.max(evals)) if evals.size else 0.0
    if emax <= 0.0:
        return np.zeros((0, q), dtype=np.float64)
    thr = emax * (np.finfo(np.float64).eps ** 0.66)
    mask = evals > thr
    if not np.any(mask):
        return np.zeros((0, q), dtype=np.float64)
    vals = np.asarray(evals[mask], dtype=np.float64)
    V_sel = np.asarray(evecs[:, mask], dtype=np.float64)
    sqrt_evals = np.sqrt(np.maximum(vals, 0.0))
    return (sqrt_evals[:, np.newaxis] * V_sel.T).astype(np.float64, copy=False)


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
        (q×q embed of ``(R + eps*I)^{-1}``), ``householder_mixing_obs_coef`` (n×q K),
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
    if np.any(w < 0):
        raise NotImplementedError("stacked QR PLS here supports only non-negative weights.")

    P = np.asarray(P, dtype=np.float64)
    penalty_sqrt, penalty_rank_template = penalty_sqrt_rows_prefer_diagonal(P)
    n_penalty_rows = int(penalty_sqrt.shape[0])
    if n_penalty_rows == 0:
        raise ValueError("penalty_sqrt_rows returned empty factor (zero penalty matrix).")

    if penalty_blocks is not None and n_coef is not None:
        penalty_rank_rows = balanced_penalty_template_sqrt_for_rank(
            penalty_blocks, fit_intercept=fit_intercept, n_coef=int(n_coef)
        )
    else:
        penalty_rank_rows = np.asarray(penalty_rank_template, dtype=np.float64)
    n_rank_penalty_rows = int(penalty_rank_rows.shape[0])
    if n_rank_penalty_rows == 0:
        raise ValueError("penalty rank sqrt has zero rows (empty penalty design).")

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
    weighted_X = outcome.weighted_X
    sqrt_w = outcome.sqrt_w
    penalty_sqrt = outcome.penalty_sqrt
    n_wx_econ = outcome.n_wx_econ
    system_rank = outcome.system_rank
    upper_r_final = outcome.upper_r_final
    qr_wx = outcome.qr_wx
    tau_wx = outcome.tau_wx
    qr_aug = outcome.qr_aug
    tau_aug = outcome.tau_aug
    pivot_aug = outcome.pivot_aug
    kept_original_indices = outcome.kept_original_indices
    dropped_column_indices = outcome.dropped_column_indices
    penalty_quadratic = outcome.penalty_quadratic

    eta = X @ coef_full

    XtWX = X.T @ (w[:, None] * X)
    A = XtWX + P

    log_det_XtWX_plus_penalty = 2.0 * float(
        np.sum(np.log(np.abs(np.diag(upper_r_final))))
    )

    sqrt_w_X = sqrt_w[:, None] * X
    if str(coef_method).lower().strip() == "lstsq":
        chol_inv_embedded = np.linalg.pinv(A, hermitian=True, rcond=1e-12)
        householder_mixing = np.zeros((n_obs, n_coef_total), dtype=np.float64)
        coef_hat_matrix = chol_inv_embedded @ XtWX
    else:
        ridge_eps = _ridge_eps_for_upper_r(upper_r_final)
        inv_upper_r = solve_triangular(
            upper_r_final + ridge_eps * np.eye(system_rank),
            np.eye(system_rank),
            lower=False,
        )
        Q_weighted_x = _dorgqr_economic(qr_wx.copy(), tau_wx, n_wx_econ)
        Q_augmented = _dorgqr_economic(qr_aug.copy(), tau_aug, system_rank)
        K_partial = Q_weighted_x @ Q_augmented[:n_wx_econ, :]

        chol_inv_embedded = np.zeros((n_coef_total, n_coef_total), dtype=np.float64)
        householder_mixing = np.zeros((n_obs, n_coef_total), dtype=np.float64)
        for i in range(system_rank):
            ki = kept_original_indices[int(pivot_aug[i])]
            householder_mixing[:, ki] = K_partial[:, i]
            for j in range(system_rank):
                kj = kept_original_indices[int(pivot_aug[j])]
                chol_inv_embedded[ki, kj] = inv_upper_r[i, j]

        coef_hat_matrix = (chol_inv_embedded @ householder_mixing.T) @ sqrt_w_X

    return {
        "coef_full": coef_full,
        "eta": eta,
        "penalty_quadratic": penalty_quadratic,
        "XtWX": XtWX,
        "A": A,
        "WX_sqrt": weighted_X,
        "E": penalty_sqrt,
        "XtWX_plus_penalty_chol_inverse_embedded": chol_inv_embedded,
        "householder_mixing_obs_coef": householder_mixing,
        "log_det_XtWX_plus_penalty": log_det_XtWX_plus_penalty,
        "penalized_system_rank": int(system_rank),
        "coef_hat_matrix": coef_hat_matrix,
        "dropped_column_indices": dropped_column_indices,
    }


def gaussian_design_needs_stacked_qr_fit(model) -> bool:
    for tb in getattr(model, "term_blocks_", None) or []:
        if str(getattr(tb, "term_type", "")).lower() == "random_effect":
            return True
    Z = getattr(model, "Z", None)
    if Z is None:
        return False
    from ..penalized_system import build_full_design

    X = np.asarray(
        build_full_design(Z, fit_intercept=bool(getattr(model, "fit_intercept", False))),
        dtype=np.float64,
    )
    if X.ndim != 2 or X.shape[1] == 0:
        return False
    return np.linalg.matrix_rank(X) < X.shape[1]
