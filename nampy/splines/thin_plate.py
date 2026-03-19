# splines/thin_plate.py
import math
import warnings

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance_matrix

from .spline_utils import eta, scale_penalty, tp_T


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
            raise ValueError("All supplied knot coordinate arrays must have the same length.")
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

    Xu = np.unique(X_shifted, axis=0)

    max_knots = int(max_knots)
    seed = int(seed)

    if Xu.shape[0] > max_knots:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Xu.shape[0], size=max_knots, replace=False)
        Xu = Xu[np.sort(idx)]

    return np.asarray(Xu, dtype=np.float64)


def _top_eigensystem(E, k):
    """
    Top-k eigensystem of a symmetric matrix E, ordered descending.
    """
    E = np.asarray(E, dtype=np.float64)
    n = E.shape[0]
    k = int(k)

    if k >= n:
        evals, U = np.linalg.eigh(E)
        idx = np.argsort(evals)[::-1][:k]
        return evals[idx], U[:, idx]

    evals, U = eigsh(E, k=k, which="LA")
    idx = np.argsort(evals)[::-1]
    return evals[idx], U[:, idx]


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

    M = null_space_dimension(d, penalty_order)

    k = int(k)
    if k < 0:
        k = default_tprs_k(d, M)
    if k < M + 1:
        warnings.warn("basis dimension k increased to the minimum possible M + 1.")
        k = M + 1

    Xu = choose_tprs_setup_locations(
        X_shifted,
        knots=setup_locations,
        max_knots=max_knots,
        seed=seed,
    )
    Xu = np.unique(np.asarray(Xu, dtype=np.float64), axis=0)

    if Xu.shape[0] < k:
        raise ValueError(
            f"tp/ts basis dimension k={k} is too large for the available "
            f"basis-setup locations ({Xu.shape[0]} unique locations)."
        )

    E = distance_matrix(Xu, Xu)
    E = eta(E, penalty_order, d)

    evals, U = _top_eigensystem(E, k)
    D = np.diag(evals)

    T = tp_T(Xu, M, penalty_order, d)
    q, _ = np.linalg.qr(U.T @ T, mode="complete")
    Z = q[:, M:]

    UZ_pen = U @ Z

    S = Z.T @ D @ Z
    S_full = np.zeros((k, k), dtype=np.float64)
    S_full[: k - M, : k - M] = S

    UZ_full = np.zeros((Xu.shape[0] + M, k), dtype=np.float64)
    UZ_full[: Xu.shape[0], : k - M] = UZ_pen
    UZ_full[Xu.shape[0] :, k - M :] = np.eye(M, dtype=np.float64)

    X_raw = thin_plate_raw_model_matrix(
        X_shifted,
        Xu,
        penalty_order,
        UZ_full,
    )

    # mgcv-style scaling step: rescale basis and penalty together
    # so smoothing parameters act on a sensible scale.
    w = np.sqrt(np.sum(X_raw**2, axis=0) / max(float(n), 1.0))
    w = np.where(w <= 0.0, 1.0, w)
    W = np.diag(1.0 / w)

    X_raw = X_raw @ W
    S_full = W @ S_full @ W
    UZ_full = UZ_full @ W

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

from dataclasses import dataclass


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


def _parse_tprs_xt(xt):
    max_knots = 2000
    seed = 1

    if xt is None:
        return max_knots, seed

    if not isinstance(xt, dict):
        raise NotImplementedError(
            "For bs='tp'/'ts', xt must currently be None or a dict "
            "with optional keys {'max.knots', 'seed'}."
        )

    if xt.get("max.knots", None) is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed", None) is not None:
        seed = int(xt["seed"])

    return max_knots, seed


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

    max_knots, seed = _parse_tprs_xt(xt)

    shift = np.mean(X, axis=0)
    X_shifted = X - shift[None, :]

    setup_knots = normalize_tprs_knots(knots, d) if knots is not None else None
    if setup_knots is not None:
        setup_knots = setup_knots - shift[None, :]

    tprs = construct_tprs_basis(
        X_shifted,
        k=k,
        penalty_order=penalty_order,
        setup_locations=setup_knots,
        max_knots=max_knots,
        seed=seed,
    )

    basis_train = thin_plate_raw_model_matrix(
        X_shifted,
        tprs["Xu"],
        penalty_order,
        tprs["UZ"],
    )

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
    )


def predict_tprs_term(X_new, setup: ThinPlateBasisSetup):
    """
    Prediction matrix from a stored ThinPlateBasisSetup.
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    X_new_shift = X_new - setup.shift[None, :]

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