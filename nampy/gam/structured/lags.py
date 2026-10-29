"""Exposure histories and distributed-lag cross-basis construction.

References
----------
- Gasparrini, A., Armstrong, B., and Kenward, M. G. (2010). Distributed lag non-linear
  models.
  https://doi.org/10.1002/sim.3940

The local history ordering, weighting and basis layout define NAMpy’s supported
distributed-lag construction.
"""

import numpy as np
import pandas as pd


def exposure_history(exposure, *, max_lag, groups=None):
    """Build float history (n,max_lag+1), ordered lag 0,1,...,max_lag.

    ``exposure`` is numeric (n,), may contain NaN but not infinity.
    ``max_lag`` is a nonnegative int. Optional ``groups`` is nonmissing (n,)
    series identifiers; each group must be contiguous. Missing leading lags
    and missing observed exposures are NaN, never zero. Rows retain input
    order and histories never cross a group boundary.
    """
    x = np.asarray(exposure, dtype=float)
    if x.ndim != 1 or not x.size or np.any(np.isinf(x)):
        raise ValueError("exposure must be a nonempty numeric vector without infinity.")
    if isinstance(max_lag, bool) or int(max_lag) != max_lag or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer.")
    max_lag = int(max_lag)
    g = np.zeros(len(x)) if groups is None else np.asarray(groups)
    if g.shape != x.shape or pd.isna(g).any():
        raise ValueError("groups must be nonmissing (n,).")
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    if pd.Series(g[starts]).duplicated().any():
        raise ValueError("Each series group must occur in one contiguous block.")
    result = np.full((len(x), max_lag + 1), np.nan)
    for a, b in zip(starts, np.r_[starts[1:], len(x)], strict=True):
        for lag in range(min(max_lag, b - a - 1) + 1):
            result[a + lag : b, lag] = x[a : b - lag]
    return result


def cross_basis(history, *, exposure_basis=None, lag_basis=None):
    """Form a dlnm-style distributed-lag nonlinear cross-basis.

    Parameters
    ----------
    history : array-like of float, shape (n,L)
        Exposure histories at the L ordered lags, e.g. exposure_history output.
    exposure_basis : callable or None, default None
        A fitted basis callback mapping flattened exposure (n*L,) to finite
        (n*L,p) basis values. None uses a linear basis x (no intercept).
        Use an already fitted basis: do not refit knots during prediction.
    lag_basis : array-like of float or None, default None
        Finite (L,q) lag basis, default ones (L,1), including a lag intercept.

    Returns
    -------
    ndarray of float, shape (n,p*q)
        Column order exposure-major, lag-basis-minor, matching dlnm.
        Each column sums exposure_basis(history_lag)*lag_basis(lag).
        Rows with any missing exposure are wholly NaN. Missing rows are not
        passed to the exposure callback. A constant exposure-basis column can
        cause rank deficiency with a model intercept; omit it by default.
    """
    H = np.asarray(history, dtype=float)
    if H.ndim != 2 or not H.size or np.any(np.isinf(H)):
        raise ValueError("history must be numeric (n,L), without infinity.")
    B = (
        np.ones((H.shape[1], 1))
        if lag_basis is None
        else np.asarray(lag_basis, dtype=float)
    )
    if (
        B.ndim != 2
        or B.shape[0] != H.shape[1]
        or not B.shape[1]
        or not np.all(np.isfinite(B))
    ):
        raise ValueError("lag_basis must be finite (L,q).")
    good = np.all(np.isfinite(H), axis=1)
    if not good.any():
        raise ValueError("At least one complete exposure history is required.")
    values = H[good].ravel()
    V = (
        values[:, None]
        if exposure_basis is None
        else np.asarray(exposure_basis(values), dtype=float)
    )
    if (
        V.ndim != 2
        or V.shape[0] != values.size
        or not V.shape[1]
        or not np.all(np.isfinite(V))
    ):
        raise ValueError("exposure_basis must return finite (n_complete*L,p).")
    V = V.reshape((-1, H.shape[1], V.shape[1]))
    result = np.full((H.shape[0], V.shape[2] * B.shape[1]), np.nan)
    result[good] = np.einsum("nlp,lq->npq", V, B).reshape((len(V), -1))
    return result


def cross_basis_penalty(
    exposure_penalty, lag_penalty, *, exposure_weight=1.0, lag_weight=1.0
):
    """Return tensor penalty (p*q,p*q) in cross_basis column order.

    Inputs are finite symmetric PSD float matrices (p,p) and (q,q).
    Nonnegative scalar weights default to 1. Result is
    exposure_weight*kron(S_exposure,I_q)+lag_weight*kron(I_p,S_lag).
    Supply to a model as a fixed parametric penalty after matching columns.
    """
    A, B = (
        np.asarray(exposure_penalty, dtype=float),
        np.asarray(lag_penalty, dtype=float),
    )
    for S in (A, B):
        if (
            S.ndim != 2
            or S.shape[0] != S.shape[1]
            or not S.size
            or not np.all(np.isfinite(S))
            or not np.allclose(S, S.T)
        ):
            raise ValueError(
                "Marginal penalties must be finite symmetric square matrices."
            )
        if np.linalg.eigvalsh(S).min() < -1e-10 * max(1, np.linalg.norm(S, 2)):
            raise ValueError("Marginal penalties must be positive semidefinite.")
    if (
        not np.all(np.isfinite([exposure_weight, lag_weight]))
        or min(exposure_weight, lag_weight) < 0
    ):
        raise ValueError("Penalty weights must be finite and nonnegative.")
    return exposure_weight * np.kron(A, np.eye(len(B))) + lag_weight * np.kron(
        np.eye(len(A)), B
    )


def cumulative_lag_matrix(
    exposure, reference, *, n_lags, exposure_basis=None, lag_basis=None
):
    """Return cumulative constant-exposure contrasts (m,p*q).

    ``exposure`` is finite (m,), ``reference`` a finite scalar and ``n_lags``
    a positive int including lag zero. Basis arguments match cross_basis.
    Contrasts sum all lags for exposure versus the same reference at every lag.
    Multiply by cross-basis coefficients for link effects, or pass as linear
    functionals with the corresponding full coefficient columns to retain
    cross-lag covariance. Exponentiation gives relative risks under log links.
    """
    x = np.asarray(exposure, dtype=float)
    if (
        x.ndim != 1
        or not x.size
        or not np.all(np.isfinite(x))
        or not np.isfinite(reference)
    ):
        raise ValueError("exposure and reference must be finite.")
    if isinstance(n_lags, bool) or int(n_lags) != n_lags or n_lags < 1:
        raise ValueError("n_lags must be a positive integer.")
    n_lags = int(n_lags)
    return cross_basis(
        np.repeat(x[:, None], n_lags, axis=1),
        exposure_basis=exposure_basis,
        lag_basis=lag_basis,
    ) - cross_basis(
        np.full((len(x), n_lags), reference),
        exposure_basis=exposure_basis,
        lag_basis=lag_basis,
    )
