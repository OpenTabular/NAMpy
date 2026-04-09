import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.spatial import distance_matrix

from .penalty_scaling import scale_penalty
from .thin_plate_basis import eta, tp_T


def _sorted_unique_rows(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.asarray(np.unique(X, axis=0), dtype=np.float64)


def _sorted_unique_inverse(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    unique, inverse = np.unique(X, axis=0, return_inverse=True)
    return np.asarray(unique, dtype=np.float64), np.asarray(inverse, dtype=int)


def _pack_covariates_colwise(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.ascontiguousarray(X.T.reshape(-1), dtype=np.float64)


def householder_qr_rowspace(A, full_q):
    """
    Python port of mgcv/src/matrix.c::QT for the n<=m case.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    Ar, Ac = A.shape
    Q = np.eye(Ac, dtype=np.float64) if full_q else np.zeros((Ar, Ac), dtype=np.float64)

    for i in range(Ar):
        p = A[i, : Ac - i].copy()
        m = float(np.max(np.abs(p))) if p.size else 0.0
        if m != 0.0:
            p /= m
        lsq = float(np.sqrt(np.dot(p, p)))
        if p.size and p[-1] < 0.0:
            lsq = -lsq
        if p.size:
            p[-1] += lsq
        g = 1.0 / (lsq * p[-1]) if lsq != 0.0 else 0.0
        lsq *= m

        for j in range(i + 1, Ar):
            x = float(np.dot(p, A[j, : Ac - i])) * g
            A[j, : Ac - i] -= x * p

        if full_q:
            for j in range(Q.shape[0]):
                x = float(np.dot(p, Q[j, : Ac - i])) * g
                Q[j, : Ac - i] -= x * p
        else:
            scale = math.sqrt(g) if g > 0.0 else 0.0
            Q[i, : Ac - i] = p * scale
            Q[i, Ac - i :] = 0.0

        A[i, Ac - i - 1] = -lsq
        A[i, : Ac - i - 1] = 0.0

    return Q, A


def apply_householder_reflectors(C, U, p, t):
    """
    Python port of mgcv/src/matrix.c::HQmult.
    """
    C = np.asarray(C, dtype=np.float64).copy()
    U = np.asarray(U, dtype=np.float64)

    if p:
        ks = range(U.shape[0]) if t else range(U.shape[0] - 1, -1, -1)
        for k in ks:
            u = U[k]
            CuV = C.T @ u
            C -= np.outer(u, CuV)
    else:
        ks = range(U.shape[0] - 1, -1, -1) if t else range(U.shape[0])
        for k in ks:
            u = U[k]
            CuV = C @ u
            C -= np.outer(CuV, u)
    return C


def construct_tprs_reference_basis(
    X_shifted,
    *,
    k,
    penalty_order,
    setup_locations=None,
    scale_columns=True,
):
    return construct_tprs_basis(
        X_shifted,
        k=k,
        penalty_order=penalty_order,
        setup_locations=setup_locations,
        scale_columns=scale_columns,
    )


def predict_tprs_reference_basis(X_new_shifted, setup):
    return thin_plate_raw_model_matrix(
        X_new_shifted,
        setup.Xu,
        setup.penalty_order,
        setup.UZ,
    )


def null_space_dimension(d: int, m: int) -> int:
    """
    mgcv::null.space.dimension-style scalar helper.

    For a thin plate spline penalty of order m in dimension d,
    the null-space dimension is:
        M = choose(m + d - 1, d)

    This helper assumes m has already been resolved/validated.
    """
    d = int(d)
    m = int(m)
    if d < 0:
        raise ValueError("d must be >= 0.")
    if m <= 0:
        raise ValueError("m must be >= 1.")
    return math.comb(m + d - 1, d)


def default_tprs_penalty_order(d: int) -> int:
    """
    mgcv default: smallest m satisfying 2m > d + 1.
    """
    d = int(d)
    m = 1
    while 2 * m <= d + 1:
        m += 1
    return m


def parse_tprs_m(m, d: int):
    """
    Parse mgcv-style tp/ts 'm' specification.

    Returns
    -------
    penalty_order : int
    drop_null : bool
        True iff m is a vector/list/tuple and its second element is 0.

    Rules followed here
    -------------------
    - m is None or <= 0: use mgcv default smallest m with 2m > d + 1
    - explicit m must satisfy 2m > d
    - m[1] == 0 requests dropping the penalty null space (tp only)
    """
    d = int(d)

    if m is None:
        return default_tprs_penalty_order(d), False

    if np.isscalar(m):
        m1 = int(m)
        drop_null = False
    else:
        vals = [int(v) for v in np.asarray(m).ravel().tolist()]
        if len(vals) == 0:
            return default_tprs_penalty_order(d), False
        m1 = vals[0]
        drop_null = len(vals) > 1 and vals[1] == 0

    if m1 <= 0:
        m1 = default_tprs_penalty_order(d)

    if 2 * m1 <= d:
        raise ValueError(
            f"Thin plate spline penalty order m={m1} is invalid for dimension d={d}. "
            "Need 2*m > d."
        )

    return m1, drop_null


def default_tprs_k(d: int, M: int) -> int:
    """
    mgcv default k = M + k.def, with k.def = 8, 27, 100 for d = 1, 2, >2.
    """
    d = int(d)
    M = int(M)
    if d <= 1:
        k_def = 8
    elif d == 2:
        k_def = 27
    else:
        k_def = 100
    return M + k_def


def normalize_tprs_knots(knots, n_dim: int):
    """
    Normalize supplied tp/ts basis-setup locations.

    Accepted forms
    --------------
    - None
    - 1D array for 1D smooths
    - 2D array of shape (n_knots, n_dim)
    - list/tuple of length n_dim containing coordinate arrays of equal length
      (mgcv-style per-variable knot vectors for one multivariate term)
    """
    if knots is None:
        return None

    n_dim = int(n_dim)

    if isinstance(knots, (list, tuple)):
        # 1D convenience
        if n_dim == 1 and len(knots) > 0 and np.isscalar(knots[0]):
            arr = np.asarray(knots, dtype=np.float64).ravel()
            return arr.reshape(-1, 1)

        if len(knots) != n_dim:
            raise ValueError(
                f"For a {n_dim}D tp/ts smooth, knots must be a 2D array or "
                f"a list/tuple of length {n_dim}."
            )

        cols = [np.asarray(k, dtype=np.float64).ravel() for k in knots]
        n = cols[0].size
        if any(c.size != n for c in cols):
            raise ValueError(
                "All supplied knot coordinate arrays must have the same length."
            )
        out = np.column_stack(cols)
        return np.asarray(out, dtype=np.float64)

    arr = np.asarray(knots, dtype=np.float64)

    if arr.ndim == 1:
        if n_dim != 1:
            raise ValueError(
                f"1D knot arrays are only valid for 1D tp/ts smooths, got n_dim={n_dim}."
            )
        return arr.reshape(-1, 1)

    if arr.ndim != 2 or arr.shape[1] != n_dim:
        raise ValueError(
            f"tp/ts knots must have shape (n_knots, {n_dim}), got {arr.shape}."
        )

    return np.asarray(arr, dtype=np.float64)


def choose_tprs_setup_locations(X_shifted, knots=None, max_knots=2000, seed=1):
    """
    mgcv-like basis-setup location handling for tp/ts.

    - If knots are supplied, use them directly as basis-setup locations.
    - Otherwise use all unique shifted covariate locations.
    - If those exceed max.knots, sample max.knots unique locations with a
      fixed seed for repeatability.
    """
    X_shifted = np.asarray(X_shifted, dtype=np.float64)
    if X_shifted.ndim == 1:
        X_shifted = X_shifted.reshape(-1, 1)

    d = X_shifted.shape[1]

    if knots is not None:
        K = normalize_tprs_knots(knots, d)
        if np.any(~np.isfinite(K)):
            raise ValueError("tp/ts knots contain NaN or Inf.")
        return np.asarray(K, dtype=np.float64)

    Xu = _sorted_unique_rows(X_shifted)

    max_knots = int(max_knots)
    seed = int(seed)

    if Xu.shape[0] > max_knots:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Xu.shape[0], size=max_knots, replace=False)
        Xu = Xu[np.sort(idx)]

    return np.asarray(Xu, dtype=np.float64)


def _top_eigensystem(E, k):
    """
    mgcv-compatible top-k eigensystem for a symmetric matrix E.

    This mirrors `Rlanczos(..., lm=-1)` in `mgcv/src/mat.c` closely enough
    to reproduce the same eigenvalue ordering and eigenvector signs used by
    `tprs_setup()`. Using a dense eigendecomposition directly gets the same
    invariant subspace, but not the same deterministic orientation that the
    parity tests require.
    """
    E = np.asarray(E, dtype=np.float64)
    n = E.shape[0]
    k = int(k)

    if k <= 0 or n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((n, 0), dtype=np.float64)
    if k > n:
        raise ValueError(f"k must be <= matrix dimension, got k={k}, n={n}.")

    tol = float(np.finfo(np.float64).eps ** 0.7)
    f_check = max(10, max(1, n // 10))

    # mgcv uses a fixed linear congruential generator to build the start vector.
    jran = 1
    ia = 106
    ic = 1283
    im = 6075
    q = []
    q0 = np.empty(n, dtype=np.float64)
    for i in range(n):
        jran = (jran * ia + ic) % im
        q0[i] = float(jran) / float(im) - 0.5
    q0 /= np.linalg.norm(q0)
    q.append(q0)

    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    err = np.full(n, 1e300, dtype=np.float64)

    d = None
    vecs = None
    j_final = n
    m_keep = k
    lm_keep = 0

    for j in range(n):
        z = E @ q[j]
        a[j] = float(q[j] @ z)

        if j == 0:
            z = z - a[j] * q[j]
        else:
            z = z - a[j] * q[j] - b[j - 1] * q[j - 1]

            # Full re-orthogonalization, repeated exactly as in mgcv.
            for i in range(j + 1):
                xx = -float(z @ q[i])
                z = z + xx * q[i]
            for i in range(j + 1):
                xx = -float(z @ q[i])
                z = z + xx * q[i]

        b[j] = float(np.linalg.norm(z))
        if j < n - 1:
            if b[j] == 0.0:
                raise np.linalg.LinAlgError(
                    "Lanczos breakdown in thin-plate eigensystem."
                )
            q.append(z / b[j])

        if ((j >= k) and (j % f_check == 0)) or (j == n - 1):
            d_asc, vecs_asc = scipy.linalg.eigh_tridiagonal(a[: j + 1], b[:j])

            # mgcv_trisymeig returns eigenvalues/eigenvectors in descending order.
            order = np.argsort(d_asc)[::-1]
            d = np.asarray(d_asc[order], dtype=np.float64)
            vecs = np.asarray(vecs_asc[:, order], dtype=np.float64)

            norm_tj = max(abs(d[0]), abs(d[j]))
            err[: j + 1] = np.abs(b[j] * vecs[-1, :])

            if j >= k:
                max_err = norm_tj * tol
                pi = 0
                ni = 0
                converged = True
                while pi + ni < k:
                    if abs(d[pi]) >= abs(d[j - ni]):
                        if err[pi] > max_err:
                            converged = False
                            break
                        pi += 1
                    else:
                        if err[ni] > max_err:
                            converged = False
                            break
                        ni += 1
                if converged:
                    m_keep = pi
                    lm_keep = ni
                    j_final = j + 1
                    break

    if d is None or vecs is None:
        raise np.linalg.LinAlgError("Failed to compute thin-plate eigensystem.")

    U = np.zeros((n, k), dtype=np.float64)
    for col in range(m_keep):
        coeff = vecs[:j_final, col]
        for q_idx in range(j_final):
            U[:, col] += q[q_idx] * coeff[q_idx]

    for col in range(m_keep, m_keep + lm_keep):
        kk = j_final - (lm_keep + m_keep - col)
        coeff = vecs[:j_final, kk]
        for q_idx in range(j_final):
            U[:, col] += q[q_idx] * coeff[q_idx]

    evals = np.zeros(k, dtype=np.float64)
    evals[:m_keep] = d[:m_keep]
    for col in range(m_keep, m_keep + lm_keep):
        kk = j_final - (lm_keep + m_keep - col)
        evals[col] = d[kk]

    return evals, U


def thin_plate_raw_model_matrix(X_shifted, Xu, penalty_order, UZ):
    """
    Evaluate the raw low-rank thin-plate regression spline model matrix.

    Parameters
    ----------
    X_shifted : array, shape (n, d)
        New or training covariate values after shift subtraction.
    Xu : array, shape (n_unique, d)
        Basis-setup locations used to construct the truncated basis.
    penalty_order : int
        Thin plate spline penalty order m.
    UZ : array
        Matrix mapping low-rank coefficients to the full thin plate spline
        coefficients.

    Returns
    -------
    B_raw : array, shape (n, k)
    """
    X_shifted = np.asarray(X_shifted, dtype=np.float64)
    Xu = np.asarray(Xu, dtype=np.float64)
    UZ = np.asarray(UZ, dtype=np.float64)

    if X_shifted.ndim == 1:
        X_shifted = X_shifted.reshape(-1, 1)
    if Xu.ndim == 1:
        Xu = Xu.reshape(-1, 1)

    d = X_shifted.shape[1]
    if Xu.shape[1] != d:
        raise ValueError(
            f"Xu has dimension {Xu.shape[1]}, but X_shifted has dimension {d}."
        )

    M = null_space_dimension(d, penalty_order)
    E = distance_matrix(X_shifted, Xu)
    E = eta(E, penalty_order, d)
    T = tp_T(X_shifted, M, penalty_order, d)
    ET = np.column_stack([E, T])
    return np.asarray(ET @ UZ, dtype=np.float64)


def construct_tprs_basis(
    X_shifted,
    *,
    k,
    penalty_order,
    setup_locations=None,
    max_knots=2000,
    seed=1,
    scale_columns=True,
):
    """
    Construct a low-rank thin-plate regression spline basis on X_shifted.

    This is an mgcv-style eigen-based tprs setup:
    - choose basis-setup locations Xu,
    - compute the truncated thin-plate basis there,
    - evaluate the resulting low-rank basis at all training locations,
    - scale basis/penalty together for numerically sensible smoothing-parameter action.

    Returns
    -------
    dict with keys:
        X_raw
        S_raw
        UZ
        Xu
        M
        k
    """
    X_shifted = np.asarray(X_shifted, dtype=np.float64)
    if X_shifted.ndim == 1:
        X_shifted = X_shifted.reshape(-1, 1)

    n, d = X_shifted.shape
    penalty_order = int(penalty_order)
    X_unique_all, inverse_all = _sorted_unique_inverse(X_shifted)

    M = null_space_dimension(d, penalty_order)

    k = int(k)
    if k < 0:
        k = default_tprs_k(d, M)
    if k < M + 1:
        warnings.warn(
            "basis dimension k increased to the minimum possible M + 1.", stacklevel=2
        )
        k = M + 1

    Xu = choose_tprs_setup_locations(
        X_shifted,
        knots=setup_locations,
        max_knots=max_knots,
        seed=seed,
    )
    Xu = _sorted_unique_rows(np.asarray(Xu, dtype=np.float64))

    xu_count = int(Xu.shape[0])
    if xu_count < M + 1:
        raise ValueError(
            f"tp/ts smooth needs at least M+1={M+1} unique locations, "
            f"got {xu_count}."
        )
    if xu_count < k:
        warnings.warn(
            f"tp/ts basis dimension k={k} reduced to available "
            f"{xu_count} unique basis-setup locations.",
            stacklevel=2,
        )
        k = xu_count

    E = distance_matrix(Xu, Xu)
    E = eta(E, penalty_order, d)

    T = tp_T(Xu, M, penalty_order, d)
    pure_knot = bool(Xu.shape[0] == k)
    if pure_knot:
        Q_full, _ = householder_qr_rowspace(T.T, full_q=True)
        Z_house, _ = householder_qr_rowspace(T.T, full_q=False)
        evals = None
        U = None
        UZ_full = np.zeros((Xu.shape[0] + M, k), dtype=np.float64)
        UZ_full[: Xu.shape[0], :] = Q_full
    else:
        evals, U = _top_eigensystem(E, k)
        TU = T.T @ U
        Z_house, _ = householder_qr_rowspace(TU, full_q=False)
        UZ_pen = apply_householder_reflectors(U.copy(), Z_house, p=0, t=0)[:, : k - M]
        UZ_full = np.zeros((Xu.shape[0] + M, k), dtype=np.float64)
        UZ_full[: Xu.shape[0], : k - M] = UZ_pen

    UZ_full[Xu.shape[0] :, k - M :] = np.eye(M, dtype=np.float64)
    no_knots_direct = (
        setup_locations is None
        and X_unique_all.shape[0] <= int(max_knots)
        and not pure_knot
    )

    if no_knots_direct:
        X1 = U * evals[np.newaxis, :]
        X1 = apply_householder_reflectors(X1, Z_house, p=0, t=0)
        X1[:, X1.shape[1] - M :] = 0.0
        X1[:, X1.shape[1] - M :] = T
        X_raw = np.asarray(X1[inverse_all, :], dtype=np.float64)
    else:
        X_raw = thin_plate_raw_model_matrix(
            X_shifted,
            Xu,
            penalty_order,
            UZ_full,
        )

    if pure_knot:
        S_full = apply_householder_reflectors(E.copy(), Z_house, p=0, t=0)
        S_full = apply_householder_reflectors(S_full, Z_house, p=1, t=1)
    else:
        S_full = np.diag(evals)
        S_full = apply_householder_reflectors(S_full, Z_house, p=0, t=0)
        S_full = apply_householder_reflectors(S_full, Z_house, p=1, t=1)
    S_full[S_full.shape[0] - M :, :] = 0.0
    S_full[:, S_full.shape[1] - M :] = 0.0

    if bool(scale_columns):
        # mgcv's tp constructor first normalizes columns to unit RMS, then
        # smoothCon(scale.penalty=TRUE) applies the usual global penalty
        # rescaling against the resulting model matrix. The final smoothing
        # parameter convention depends on both steps.
        for j in range(X_raw.shape[1]):
            w = float(np.sqrt(np.mean(X_raw[:, j] ** 2)))
            if not np.isfinite(w) or w <= 0.0:
                continue
            X_raw[:, j] /= w
            UZ_full[:, j] /= w
            S_full[j, :] /= w
            S_full[:, j] /= w
        S_full = scale_penalty(X_raw, S_full)

    S_full = 0.5 * (S_full + S_full.T)

    return {
        "X_raw": np.asarray(X_raw, dtype=np.float64),
        "S_raw": np.asarray(S_full, dtype=np.float64),
        "UZ": np.asarray(UZ_full, dtype=np.float64),
        "Xu": np.asarray(Xu, dtype=np.float64),
        "M": int(M),
        "k": int(k),
    }


def full_rank_shrinkage_penalty(S, shrink=1e-1, tol=1e-12):
    """
    mgcv::ts-style null-space shrinkage.

    Replace the zero eigenvalues of S by a small multiple of the smallest
    strictly positive eigenvalue, giving a full-rank penalty.
    """
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)

    evals, U = np.linalg.eigh(S)
    tol_eff = tol * max(1.0, np.max(np.abs(evals)) if evals.size else 1.0)

    pos = evals > tol_eff
    if not np.any(pos):
        return S.copy()

    min_pos = float(np.min(evals[pos]))
    out = evals.copy()
    out[~pos] = min_pos * float(shrink)

    S_full = (U * out) @ U.T
    return 0.5 * (S_full + S_full.T)


@dataclass
class ThinPlateBasisSetup:
    basis_name: str
    shift: np.ndarray
    Xu: np.ndarray
    UZ: np.ndarray
    penalty_order: int
    original_null_space_dim: int
    rank: int
    bs_dim: int
    basis_train: np.ndarray
    penalty: np.ndarray
    drop_null_requested: bool
    drop_null_effective: bool
    drop_keep: int | None = None
    cmX: np.ndarray | None = None
    mgcv_c_backend: bool = False


def _parse_tprs_xt(xt):
    max_knots = 2000
    seed = 1
    scale_columns = True
    no_shift = False

    if xt is None:
        return max_knots, seed, scale_columns, no_shift

    if not isinstance(xt, dict):
        raise NotImplementedError(
            "For bs='tp'/'ts', xt must currently be None or a dict "
            "with optional keys {'max.knots', 'seed'}."
        )

    if xt.get("max.knots", None) is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed", None) is not None:
        seed = int(xt["seed"])
    if xt.get("__scale_columns", None) is not None:
        scale_columns = bool(xt["__scale_columns"])
    if xt.get("__no_shift", None) is not None:
        no_shift = bool(xt["__no_shift"])

    return max_knots, seed, scale_columns, no_shift


def build_tprs_term_setup(
    X,
    *,
    basis="tp",
    k=-1,
    m=None,
    knots=None,
    xt=None,
):
    """
    High-level thin-plate/shrinkage-thin-plate setup for term classes.

    Returns a ThinPlateBasisSetup containing everything required for both
    training and prediction, while keeping the term class free of basis
    construction details.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    basis = str(basis).lower()
    if basis not in {"tp", "ts"}:
        raise ValueError(f"basis must be 'tp' or 'ts', got {basis!r}.")

    d = X.shape[1]
    penalty_order, drop_null = parse_tprs_m(m, d)
    original_null_space_dim = null_space_dimension(d, penalty_order)

    max_knots, seed, scale_columns, no_shift = _parse_tprs_xt(xt)

    shift = np.zeros(X.shape[1], dtype=np.float64) if no_shift else np.mean(X, axis=0)
    X_shifted = X - shift[None, :]

    setup_knots = normalize_tprs_knots(knots, d) if knots is not None else None
    if setup_knots is not None:
        setup_knots = setup_knots - shift[None, :]
        if setup_knots.shape[0] > X.shape[0]:
            warnings.warn(
                "more knots than data in a tp term: knots ignored.",
                stacklevel=2,
            )
            setup_knots = None

    bs_dim = int(k)
    if bs_dim < 0:
        bs_dim = default_tprs_k(d, original_null_space_dim)
    if bs_dim < original_null_space_dim + 1:
        warnings.warn(
            "basis dimension k increased to the minimum possible M + 1.", stacklevel=2
        )
        bs_dim = original_null_space_dim + 1

    mgcv_setup_locations = setup_knots
    if mgcv_setup_locations is None and X.shape[0] > max_knots:
        Xu_all = np.unique(X_shifted, axis=0)
        if Xu_all.shape[0] > max_knots:
            mgcv_setup_locations = choose_tprs_setup_locations(
                X_shifted,
                knots=None,
                max_knots=max_knots,
                seed=seed,
            )

    tprs = construct_tprs_reference_basis(
        X_shifted,
        k=bs_dim,
        penalty_order=penalty_order,
        setup_locations=mgcv_setup_locations,
        scale_columns=scale_columns,
    )
    mgcv_c_backend = tprs is not None

    if tprs is None:
        tprs = construct_tprs_basis(
            X_shifted,
            k=bs_dim,
            penalty_order=penalty_order,
            setup_locations=setup_knots,
            max_knots=max_knots,
            seed=seed,
            scale_columns=scale_columns,
        )

    basis_train = np.asarray(tprs["X_raw"], dtype=np.float64)

    penalty = np.asarray(tprs["S_raw"], dtype=np.float64)
    original_null_space_dim = int(tprs["M"])
    bs_dim = int(tprs["k"])

    drop_null_effective = bool(drop_null)
    drop_keep = None
    cmX = None

    if basis == "ts":
        penalty = full_rank_shrinkage_penalty(penalty, shrink=1e-1)
        drop_null_effective = False

    if drop_null_effective:
        drop_keep = bs_dim - original_null_space_dim
        basis_train = basis_train[:, :drop_keep]
        penalty = penalty[:drop_keep, :drop_keep]

        # mgcv-style centering after dropping the null space
        cmX = np.mean(basis_train, axis=0)
        basis_train = basis_train - cmX[None, :]

    rank = int(np.linalg.matrix_rank(penalty))

    return ThinPlateBasisSetup(
        basis_name=basis,
        shift=np.asarray(shift, dtype=np.float64),
        Xu=np.asarray(tprs["Xu"], dtype=np.float64),
        UZ=np.asarray(tprs["UZ"], dtype=np.float64),
        penalty_order=int(penalty_order),
        original_null_space_dim=original_null_space_dim,
        rank=rank,
        bs_dim=int(basis_train.shape[1]),
        basis_train=np.asarray(basis_train, dtype=np.float64),
        penalty=np.asarray(0.5 * (penalty + penalty.T), dtype=np.float64),
        drop_null_requested=bool(drop_null),
        drop_null_effective=bool(drop_null_effective),
        drop_keep=(None if drop_keep is None else int(drop_keep)),
        cmX=(None if cmX is None else np.asarray(cmX, dtype=np.float64)),
        mgcv_c_backend=bool(mgcv_c_backend),
    )


def predict_tprs_term(X_new, setup: ThinPlateBasisSetup):
    """
    Prediction matrix from a stored ThinPlateBasisSetup.
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    X_new_shift = X_new - setup.shift[None, :]

    B = None
    if bool(getattr(setup, "mgcv_c_backend", False)):
        B = predict_tprs_reference_basis(X_new_shift, setup)

    if B is None:
        B = thin_plate_raw_model_matrix(
            X_new_shift,
            setup.Xu,
            setup.penalty_order,
            setup.UZ,
        )

    if setup.drop_null_effective:
        B = B[:, : setup.drop_keep]
        if setup.cmX is not None:
            B = B - setup.cmX[None, :]

    return np.asarray(B, dtype=np.float64)
