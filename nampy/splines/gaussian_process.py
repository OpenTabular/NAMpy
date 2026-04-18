import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance_matrix

from .._column_orientation import apply_column_signs, canonical_column_signs
from .thin_plate import _top_eigensystem


def normalize_gp_knots(knots, n_dim: int):
    """
    Normalize supplied gp basis-setup locations.

    Accepted forms
    --------------
    - None
    - 1D array for 1D smooths
    - 2D array of shape (n_knots, n_dim)
    - list/tuple of length n_dim containing coordinate arrays of equal length
    """
    if knots is None:
        return None

    n_dim = int(n_dim)

    if isinstance(knots, (list, tuple)):
        if n_dim == 1 and len(knots) > 0 and np.isscalar(knots[0]):
            arr = np.asarray(knots, dtype=np.float64).ravel()
            return arr.reshape(-1, 1)

        if len(knots) != n_dim:
            raise ValueError(
                f"For a {n_dim}D gp smooth, knots must be a 2D array or "
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
                f"1D knot arrays are only valid for 1D gp smooths, got n_dim={n_dim}."
            )
        return arr.reshape(-1, 1)

    if arr.ndim != 2 or arr.shape[1] != n_dim:
        raise ValueError(
            f"gp knots must have shape (n_knots, {n_dim}), got {arr.shape}."
        )

    return np.asarray(arr, dtype=np.float64)


def parse_gp_m(m):
    """
    Parse mgcv-style gp 'm' specification.

    Returns
    -------
    gp_type : int
        1..5 for spherical, power exponential, Matern 1.5/2.5/3.5
    stationary : bool
        True iff m[1] < 0
    rho : float
        <= 0 means "choose automatically"
    power : float
        Used only for the power exponential family
    """
    if m is None:
        return 3, False, -1.0, 1.0

    vals = np.asarray(m, dtype=np.float64).ravel()
    if vals.size == 0 or (vals.size == 1 and np.isnan(vals[0])):
        return 3, False, -1.0, 1.0

    m1 = float(vals[0])
    gp_type = int(abs(np.round(m1)))
    stationary = bool(m1 < 0.0)

    rho = float(vals[1]) if vals.size > 1 else -1.0
    power = float(vals[2]) if vals.size > 2 else 1.0

    if gp_type not in {1, 2, 3, 4, 5}:
        raise ValueError("For bs='gp', abs(m[0]) must be one of {1,2,3,4,5}.")
    if gp_type == 2:
        if not (0.0 < power <= 2.0):
            raise ValueError(
                "For bs='gp' power exponential, m[2] must satisfy 0 < power <= 2."
            )
    else:
        power = 1.0

    return gp_type, stationary, rho, power


def gp_polynomial_tail_basis(x_shifted, gp_defn):
    """
    Polynomial tail basis for GP smooths.
    """
    x_shifted = np.asarray(x_shifted, dtype=np.float64)
    if x_shifted.ndim == 1:
        x_shifted = x_shifted.reshape(-1, 1)

    stationary = bool(gp_defn["stationary"])
    n = x_shifted.shape[0]

    if stationary:
        return np.ones((n, 1), dtype=np.float64)
    return np.column_stack([np.ones(n, dtype=np.float64), x_shifted])


def gp_kernel_matrix(x, xk, gp_defn):
    """
    GP kernel matrix between locations.
    """
    x = np.asarray(x, dtype=np.float64)
    xk = np.asarray(xk, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if xk.ndim == 1:
        xk = xk.reshape(-1, 1)

    E = distance_matrix(x, xk)

    rho = float(gp_defn["rho"])
    if rho <= 0.0:
        rho = float(np.max(E))
        if rho <= 0.0:
            rho = 1.0

    gp_type = int(gp_defn["type"])
    power = float(gp_defn["power"])

    D = E / rho

    if gp_type == 1:
        K = (1.0 - 1.5 * D + 0.5 * D**3) * (D <= 1.0)
    elif gp_type == 2:
        K = np.exp(-(D**power))
    elif gp_type == 3:
        eD = np.exp(-D)
        K = (1.0 + D) * eD
    elif gp_type == 4:
        eD = np.exp(-D)
        K = eD + (D * eD) * (1.0 + D / 3.0)
    elif gp_type == 5:
        eD = np.exp(-D)
        K = eD + (D * eD) * (1.0 + 0.4 * D + D**2 / 15.0)
    else:
        raise ValueError(f"Unknown gp correlation type {gp_type}.")

    resolved = {
        "type": gp_type,
        "stationary": bool(gp_defn["stationary"]),
        "rho": float(rho),
        "power": float(power),
    }
    return np.asarray(K, dtype=np.float64), resolved


def default_gp_bs_dim(d: int) -> int:
    """
    Follow current mgcv source behavior for default full basis dimension:
        d + 1 + c(10,30,100)[d]
    for d = 1, 2, >2.
    """
    d = int(d)
    if d <= 1:
        return d + 1 + 10
    if d == 2:
        return d + 1 + 30
    return d + 1 + 100


def choose_gp_setup_locations(
    X_shifted, knots=None, n_rows=None, max_knots=2000, seed=1
):
    """
    mgcv-like basis-setup location handling for bs='gp'.

    - If knots are supplied, use them directly unless there are more supplied
      locations than observations, in which case ignore them with a warning.
    - Otherwise use all unique shifted covariate locations.
    - If unique locations exceed max.knots, sample max.knots with a fixed seed.
    """
    X_shifted = np.asarray(X_shifted, dtype=np.float64)
    if X_shifted.ndim == 1:
        X_shifted = X_shifted.reshape(-1, 1)

    d = X_shifted.shape[1]
    n_rows = X_shifted.shape[0] if n_rows is None else int(n_rows)

    if knots is not None:
        K = normalize_gp_knots(knots, d)
        if np.any(~np.isfinite(K)):
            raise ValueError("gp knots contain NaN or Inf.")
        if K.shape[0] > n_rows:
            warnings.warn(
                "more knots than data in a gp term: knots ignored.", stacklevel=2
            )
        else:
            return np.asarray(K, dtype=np.float64)

    Xu = np.unique(X_shifted, axis=0)

    max_knots = int(max_knots)
    seed = int(seed)

    if n_rows > max_knots and Xu.shape[0] > max_knots:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Xu.shape[0], size=max_knots, replace=False)
        Xu = Xu[np.sort(idx)]

    return np.asarray(Xu, dtype=np.float64)


def gp_setup_from_data(
    X_shifted,
    *,
    bs_dim,
    m=None,
    knots=None,
    max_knots=2000,
    seed=1,
):
    """
    Construct the low-rank gp smooth setup.

    Returns
    -------
    dict with:
        gp_defn
        shiftless_knots
        UZ
        penalty
        null_space_dim
        rank
        bs_dim
    """
    X_shifted = np.asarray(X_shifted, dtype=np.float64)
    if X_shifted.ndim == 1:
        X_shifted = X_shifted.reshape(-1, 1)

    n, d = X_shifted.shape
    xu = np.unique(X_shifted, axis=0)

    if bs_dim is None or int(bs_dim) < 0:
        bs_dim = default_gp_bs_dim(d)
    bs_dim = int(bs_dim)

    # Follow current mgcv source minimum.
    if bs_dim < d + 2:
        warnings.warn("basis dimension reset to minimum possible", stacklevel=2)
        bs_dim = d + 2

    if xu.shape[0] < bs_dim:
        raise ValueError(
            "A term has fewer unique covariate combinations than specified maximum "
            "degrees of freedom for bs='gp'."
        )

    gp_type, stationary, rho, power = parse_gp_m(m)
    null_space_dim = 1 if stationary else d + 1
    rank = bs_dim - null_space_dim
    if rank < 1:
        raise ValueError("bs='gp' requires bs_dim to exceed the null-space dimension.")

    knt = choose_gp_setup_locations(
        X_shifted,
        knots=knots,
        n_rows=n,
        max_knots=max_knots,
        seed=seed,
    )
    nk = knt.shape[0]

    if nk < rank:
        raise ValueError(
            f"bs='gp' has {nk} basis-setup locations but requires penalized rank {rank}. "
            "Reduce k or provide more setup knots."
        )

    E, resolved = gp_kernel_matrix(
        knt,
        knt,
        {"type": gp_type, "stationary": stationary, "rho": rho, "power": power},
    )

    if rank < nk:
        # mgcv::smooth.construct.gp.smooth.spec uses slanczos(E, k, -1),
        # i.e. the same largest-magnitude truncated eigensystem used by tprs.
        evals, UZ = _top_eigensystem(E, rank)
        penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
        penalty[:rank, :rank] = np.diag(evals)
    else:
        # direct knot-basis case
        UZ = np.eye(nk, dtype=np.float64)
        penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
        penalty[:rank, :rank] = E

    col_signs = canonical_column_signs(UZ)
    UZ = apply_column_signs(UZ, col_signs)

    penalty = 0.5 * (penalty + penalty.T)

    return {
        "gp_defn": resolved,
        "knt": np.asarray(knt, dtype=np.float64),
        "UZ": np.asarray(UZ, dtype=np.float64),
        "penalty": penalty,
        "null_space_dim": int(null_space_dim),
        "rank": int(rank),
        "bs_dim": int(bs_dim),
    }


def gp_predict_matrix(x_shifted, *, knt, UZ, gp_defn):
    """
    mgcv-like GP prediction matrix:
        cbind(kernel(x, knt, gp.defn) %*% UZ, polynomial_tail(x, gp.defn))
    """
    x_shifted = np.asarray(x_shifted, dtype=np.float64)
    if x_shifted.ndim == 1:
        x_shifted = x_shifted.reshape(-1, 1)

    knt = np.asarray(knt, dtype=np.float64)
    UZ = np.asarray(UZ, dtype=np.float64)

    E, _ = gp_kernel_matrix(x_shifted, knt, gp_defn)
    T = gp_polynomial_tail_basis(x_shifted, gp_defn)
    return np.column_stack([E @ UZ, T])


@dataclass
class GPBasisSetup:
    shift: np.ndarray
    knt: np.ndarray
    UZ: np.ndarray
    gp_defn: dict
    null_space_dim: int
    rank: int
    bs_dim: int
    basis_train: np.ndarray
    penalty: np.ndarray


def _parse_gp_xt(xt):
    max_knots = 2000
    seed = 1

    if xt is None:
        return max_knots, seed

    if not isinstance(xt, dict):
        raise NotImplementedError(
            "For bs='gp', xt must currently be None or a dict with optional "
            "keys {'max.knots', 'seed'}."
        )

    if xt.get("max.knots", None) is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed", None) is not None:
        seed = int(xt["seed"])

    return max_knots, seed


def build_gp_term_setup(
    X,
    *,
    k=-1,
    m=None,
    knots=None,
    xt=None,
):
    """
    High-level Gaussian-process smooth setup for term classes.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    max_knots, seed = _parse_gp_xt(xt)

    shift = np.mean(X, axis=0)
    X_shifted = X - shift[None, :]

    setup = gp_setup_from_data(
        X_shifted,
        bs_dim=k,
        m=m,
        knots=knots,
        max_knots=max_knots,
        seed=seed,
    )

    basis_train = gp_predict_matrix(
        X_shifted,
        knt=setup["knt"],
        UZ=setup["UZ"],
        gp_defn=setup["gp_defn"],
    )

    return GPBasisSetup(
        shift=np.asarray(shift, dtype=np.float64),
        knt=np.asarray(setup["knt"], dtype=np.float64),
        UZ=np.asarray(setup["UZ"], dtype=np.float64),
        gp_defn=dict(setup["gp_defn"]),
        null_space_dim=int(setup["null_space_dim"]),
        rank=int(setup["rank"]),
        bs_dim=int(setup["bs_dim"]),
        basis_train=np.asarray(basis_train, dtype=np.float64),
        penalty=np.asarray(
            0.5 * (setup["penalty"] + setup["penalty"].T), dtype=np.float64
        ),
    )


def predict_gp_term(X_new, setup: GPBasisSetup):
    """
    Prediction matrix from a stored GPBasisSetup.
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    X_new_shift = X_new - setup.shift[None, :]

    return np.asarray(
        gp_predict_matrix(
            X_new_shift,
            knt=setup.knt,
            UZ=setup.UZ,
            gp_defn=setup.gp_defn,
        ),
        dtype=np.float64,
    )
