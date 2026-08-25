"""Natural-parameterization helpers shared by smooth constructors."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from ...linalg import matrix_is_rank_deficient


def _r_symmetric_eigh_descending(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror base R's ``eigen(x, symmetric=TRUE)`` value ordering.

    Eigenvectors within a repeated eigenspace are not uniquely identified and
    can differ across the BLAS/LAPACK libraries used by R and SciPy.  Exact raw
    vector parity up to column sign is therefore only expected for simple
    eigenvalues; repeated blocks must be compared through their invariant span.
    """
    values, vectors = scipy_eigh(
        matrix,
        lower=True,
        check_finite=False,
    )
    return (
        np.asarray(values[::-1], dtype=np.float64),
        np.asarray(vectors[:, ::-1], dtype=np.float64),
    )


def _r_triangular_solve(
    triangle: np.ndarray,
    rhs: np.ndarray,
    *,
    lower: bool,
) -> np.ndarray:
    """Mirror the reference-BLAS DTRSM used by base R's triangular solves."""
    a = np.array(triangle, dtype=np.float64, order="F", copy=True)
    b = np.array(rhs, dtype=np.float64, order="F", copy=True)
    rows, columns = b.shape
    if a.shape != (rows, rows):
        raise ValueError("Triangular solve requires a square left-hand matrix.")

    # R/src/main/array.c::do_backsolve calls DTRSM with side='L',
    # transa='N', diag='N', alpha=1. Follow the Netlib operation order.
    if lower:
        indices = range(rows)
    else:
        indices = range(rows - 1, -1, -1)
    for column in range(columns):
        for pivot in indices:
            if b[pivot, column] == 0.0:
                continue
            b[pivot, column] /= a[pivot, pivot]
            value = b[pivot, column]
            affected = range(pivot + 1, rows) if lower else range(pivot)
            for row in affected:
                b[row, column] -= value * a[row, pivot]
    return np.asarray(b, dtype=np.float64)


def _r_linpack_qr(X: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the thin Q and R produced by base R's default ``qr`` path.

    This mirrors R's ``dqrdc2`` followed by ``qr.R`` and ``qr.Q``.  The
    distinction from LAPACK QR is behaviorally relevant here: ``smoothCon``
    subsequently scales penalties using an absolute row-sum norm of the
    reparameterized model matrix.
    """
    qr = np.array(X, dtype=np.float64, order="F", copy=True)
    n, p = qr.shape
    if n < p:
        raise ValueError(
            "Model matrix cannot be full column rank when it has fewer rows "
            "than columns."
        )

    qraux = np.empty(p, dtype=np.float64)
    work_original = np.empty(p, dtype=np.float64)
    work_scale = np.empty(p, dtype=np.float64)
    pivot = np.arange(p)

    # R/src/appl/dqrdc2.f: initial column norms.
    for column in range(p):
        norm = float(np.linalg.norm(qr[:, column])) if n else 0.0
        qraux[column] = norm
        work_original[column] = norm
        work_scale[column] = norm if norm != 0.0 else 1.0

    lup = min(n, p)
    rank_boundary = p
    for column in range(lup):
        # Move columns judged negligible to the right, preserving their order.
        while (
            column < rank_boundary
            and qraux[column] < work_scale[column] * tol
        ):
            moved_column = qr[:, column].copy()
            moved_pivot = pivot[column]
            moved_qraux = qraux[column]
            moved_original = work_original[column]
            moved_scale = work_scale[column]
            qr[:, column:-1] = qr[:, column + 1 :]
            qr[:, -1] = moved_column
            pivot[column:-1] = pivot[column + 1 :]
            pivot[-1] = moved_pivot
            qraux[column:-1] = qraux[column + 1 :]
            qraux[-1] = moved_qraux
            work_original[column:-1] = work_original[column + 1 :]
            work_original[-1] = moved_original
            work_scale[column:-1] = work_scale[column + 1 :]
            work_scale[-1] = moved_scale
            rank_boundary -= 1

        if column == n - 1:
            continue

        tail = qr[column:, column]
        norm = float(np.linalg.norm(tail))
        if norm == 0.0:
            continue
        if qr[column, column] != 0.0:
            norm = float(np.copysign(norm, qr[column, column]))
        qr[column:, column] = tail * (1.0 / norm)
        qr[column, column] += 1.0

        for following in range(column + 1, p):
            vector = qr[column:, column]
            target = qr[column:, following]
            multiplier = -float(vector @ target) / qr[column, column]
            qr[column:, following] = target + multiplier * vector
            if qraux[following] != 0.0:
                reduction = 1.0 - (
                    abs(qr[column, following]) / qraux[following]
                ) ** 2
                reduction = max(reduction, 0.0)
                if abs(reduction) >= 1e-6:
                    qraux[following] *= np.sqrt(reduction)
                else:
                    qraux[following] = float(
                        np.linalg.norm(qr[column + 1 :, following])
                    )
                    work_original[following] = qraux[following]

        qraux[column] = qr[column, column]
        qr[column, column] = -norm

    qr_rank = min(rank_boundary, n)
    if qr_rank < p or not np.array_equal(pivot, np.arange(p)):
        raise ValueError(
            "Model matrix is not full rank in natural-parameter construction."
        )

    R = np.triu(np.asarray(qr[:p, :], dtype=np.float64))

    # R/src/appl/dqrsl.f, job=10000: apply the stored Householder vectors in
    # reverse order to the first p columns of the identity to obtain Q.
    Q = np.array(np.eye(n, p, dtype=np.float64), order="F", copy=True)
    for column in range(min(qr_rank, n - 1) - 1, -1, -1):
        if qraux[column] == 0.0:
            continue
        vector = qr[column:, column].copy()
        vector[0] = qraux[column]
        for q_column in range(p):
            target = Q[column:, q_column]
            multiplier = -float(vector @ target) / vector[0]
            Q[column:, q_column] = target + multiplier * vector

    return Q, R


def nat_param_type1(X, S, rank=None, tol=None, unit_fnorm=True):
    """
    Python implementation of ``mgcv::nat.param(X, S, type=1)``.

    This reparameterizes so that the penalty in the penalized columns is the
    identity. Returns a dict with transformed model matrix ``X``, positive
    diagonal penalty entries ``D``, coefficient back-transform ``P``, and
    penalty rank.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tol = np.finfo(float).eps**0.8 if tol is None else float(tol)

    # Mirror mgcv/R/smooth.r::nat.param(type=1): base R QR followed by
    # eigen(RSR, symmetric=TRUE). Do not explicitly symmetrize RSR: the triangle
    # selected by base R's DSYEVR call is part of the upstream operation path.
    # Repeated null-eigenspace orientation can nevertheless remain dependent on
    # the linked BLAS/LAPACK implementation; see
    # ``_r_symmetric_eigh_descending``.
    Q, R = _r_linpack_qr(X, tol)
    if matrix_is_rank_deficient(R):
        raise ValueError(
            "Model matrix is not full rank in natural-parameter construction."
        )

    tmp = _r_triangular_solve(R.T, S.T, lower=True)
    RSR = _r_triangular_solve(R.T, tmp.T, lower=True)
    evals, U = _r_symmetric_eigh_descending(RSR)

    if rank is None or rank < 1 or rank > S.shape[0]:
        max_eval = np.max(evals) if evals.size else 0.0
        thresh = max_eval * tol
        rank = int(np.sum(evals > thresh))
    rank = max(0, min(rank, S.shape[0]))

    D = evals[:rank].copy()
    Xn = Q @ U
    P = _r_triangular_solve(R, U, lower=False)

    total_cols = Xn.shape[1]
    E = np.ones(total_cols, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(D)
    Xn = Xn / E[np.newaxis, :]
    P = P / E[np.newaxis, :]
    D = np.ones(rank, dtype=np.float64)

    if unit_fnorm:
        if rank > 0:
            scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
            Xn[:, :rank] *= scale
            P[:, :rank] *= scale
            D *= scale**2

        if rank < Xn.shape[1]:
            scalef = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
            Xn[:, rank:] *= scalef
            P[:, rank:] *= scalef

    return {
        "X": Xn,
        "D": D,
        "P": P,
        "rank": int(rank),
    }


def nat_param_type0(X, S, rank=None, tol=None, unit_fnorm=True):
    """Python implementation of ``mgcv::nat.param(X, S, type=0)``.

    Unlike type 1, the positive natural-parameter penalty eigenvalues are
    retained rather than normalized to one.  MRF reduced-rank construction
    uses this exact parameterization before keeping the least penalized
    columns.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tol = np.finfo(float).eps**0.8 if tol is None else float(tol)

    Q, R = _r_linpack_qr(X, tol)
    if matrix_is_rank_deficient(R):
        raise ValueError(
            "Model matrix is not full rank in natural-parameter construction."
        )

    tmp = _r_triangular_solve(R.T, S.T, lower=True)
    RSR = _r_triangular_solve(R.T, tmp.T, lower=True)
    evals, U = _r_symmetric_eigh_descending(RSR)

    if rank is None or rank < 1 or rank > S.shape[0]:
        max_eval = np.max(evals) if evals.size else 0.0
        rank = int(np.sum(evals > max_eval * tol))
    rank = max(0, min(int(rank), S.shape[0]))

    D = np.asarray(evals[:rank], dtype=np.float64).copy()
    Xn = np.asarray(Q @ U, dtype=np.float64)
    P = _r_triangular_solve(R, U, lower=False)

    if unit_fnorm:
        if rank > 0:
            scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
            Xn[:, :rank] *= scale
            P[:, :rank] *= scale
            D *= scale**2
        if rank < Xn.shape[1]:
            scalef = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
            Xn[:, rank:] *= scalef
            P[:, rank:] *= scalef

    return {"X": Xn, "D": D, "P": P, "rank": int(rank)}


def nat_param_type3(X, S, rank=None, tol=None, unit_fnorm=True):
    """Python implementation of ``mgcv::nat.param(..., type=3)``.

    Type 3 diagonalizes the supplied penalty without the QR step used by the
    type-0/1 natural parameterizations.  Its penalized columns have a common
    ridge penalty and, when the null space has dimension greater than one, the
    final null-space column is oriented toward the constant function.  This is
    the marginal parameterization used by ``mgcv::t2``.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if X.ndim != 2 or S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("nat_param_type3 requires a matrix X and square penalty S.")
    if X.shape[1] != S.shape[0]:
        raise ValueError("X and S must have the same coefficient dimension.")

    tol = np.finfo(float).eps ** 0.8 if tol is None else float(tol)
    evals, vectors = _r_symmetric_eigh_descending(S)
    if rank is None or int(rank) < 1 or int(rank) > S.shape[0]:
        largest = float(np.max(evals)) if evals.size else 0.0
        rank = int(np.sum(evals > largest * tol))
    rank = max(0, min(int(rank), S.shape[0]))
    if rank == 0:
        raise ValueError(
            "type-3 natural parameterization requires positive penalty rank."
        )

    null_exists = rank < X.shape[1]
    divisor = np.ones(X.shape[1], dtype=np.float64)
    divisor[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    Xn = np.asarray(X @ vectors, dtype=np.float64)
    col_norm = np.sum(Xn**2, axis=0) / divisor**2
    average_penalized_norm = float(np.mean(col_norm[:rank]))
    if not np.isfinite(average_penalized_norm) or average_penalized_norm <= 0.0:
        raise ValueError("Penalized type-3 marginal basis has zero norm.")
    if null_exists:
        divisor[rank:] = np.sqrt(
            np.maximum(col_norm[rank:] / average_penalized_norm, 0.0)
        )
        if np.any(divisor[rank:] <= 0.0):
            raise ValueError("Type-3 marginal null-space basis is rank deficient.")

    P = np.asarray(vectors / divisor[np.newaxis, :], dtype=np.float64)
    Xn = np.asarray(Xn / divisor[np.newaxis, :], dtype=np.float64)

    # mgcv reverses the destination null-space indices so that the eigenvector
    # of the centered null basis with smallest eigenvalue (the constant
    # direction) is placed last.
    if null_exists and rank < Xn.shape[1] - 1:
        source = np.arange(rank, Xn.shape[1], dtype=int)
        destination = source[::-1]
        centered = Xn[:, source] - np.mean(Xn[:, source], axis=0, keepdims=True)
        _null_values, null_vectors = _r_symmetric_eigh_descending(centered.T @ centered)
        X_source = Xn[:, source].copy()
        P_source = P[:, source].copy()
        Xn[:, destination] = X_source @ null_vectors
        P[:, destination] = P_source @ null_vectors

    scale = 1.0
    if unit_fnorm:
        scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
        Xn[:, :rank] *= scale
        # This is algebraically the same as mgcv's row scaling here because
        # type 3 equalizes the penalized and null block norms before this step.
        P[:, :rank] *= scale
        if null_exists:
            null_scale = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
            Xn[:, rank:] *= null_scale
            P[:, rank:] *= null_scale

    return {
        "X": np.asarray(Xn, dtype=np.float64),
        "D": np.full(rank, scale**2, dtype=np.float64),
        "P": np.asarray(P, dtype=np.float64),
        "rank": int(rank),
    }


__all__ = ["nat_param_type0", "nat_param_type1", "nat_param_type3"]
