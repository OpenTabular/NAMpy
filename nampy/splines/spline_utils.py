# splines/spline_utils.py
import bisect
import math

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance_matrix


def eta(E, m, d):
    """
    Calculate the eta function given a matrix of Euclidean distances, penalty order,
    and dimensionality of the data.

    Parameters
    ----------
    E : array-like
        Matrix of Euclidean distances.
    m : int
        Penalty order.
    d : int
        Dimensionality of the data.

    Returns
    -------
    np.ndarray
        Eta(E).
    """
    E = np.asarray(E, dtype=np.float64)

    if d % 2 == 0:
        d_half = d // 2
        const = ((-1) ** (m + 1 + d_half)) / (
            2 ** (2 * m - 1)
            * np.pi ** d_half
            * math.factorial(m - 1)
            * math.factorial(m - d_half)
        )

        out = np.zeros_like(E, dtype=np.float64)
        mask = E > 0
        out[mask] = const * (E[mask] ** (2 * m - d)) * np.log(E[mask])
        E = out
    else:
        E = (
            math.gamma(d / 2 - m)
            / (2 ** (2 * m) * np.pi ** (d / 2) * math.factorial(m - 1))
            * E ** (2 * m - d)
        )

    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


def tp_spline(x, k, pen_order, n, d, M):
    # subtract mean from data (try to recreate model matrix in mgcv, did not work.
    # Doesn't change the model, so can be ignored)
    if d == 1:
        x = x - x.mean()

    # reduce the data to unique observations and save the index to create the full matrix later
    x_un = np.unique(x, axis=0)
    map_idx = np.all(
        (np.expand_dims(np.array(x_un), 0) == np.expand_dims(x, 1)), axis=2
    )
    map_idx = np.argwhere(map_idx)

    # matrix of euclidean distances needed for eta
    E = distance_matrix(x_un, x_un)
    E = eta(E, pen_order, d)

    # get first k eigenvalues
    # eigsh because it is way faster than np.linalg.eigh
    eigen_values, U = eigsh(E, k, which="LA")
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    U = U[:, idx]
    D = np.diag(eigen_values)

    # U_k: first k eigenvectors
    # D_k: diagonal matrix of first k eigenvalues
    U_k = U[:, :k]
    D_k = D[:k, :k]
    T = tp_T(x_un, M, pen_order, d)

    # absorb constraint T * delta = 0
    q, r = np.linalg.qr(np.dot(U_k.T, T), mode="complete")
    Z_k = q[:, M:]

    UZ = U_k @ Z_k

    # create penalty matrix S (padded by zeros for unpenalized alpha-part)
    S = Z_k.T @ D_k @ Z_k
    S_full = np.zeros((k, k))
    S_full[: k - M, : k - M] = S

    # finalize design matrix
    X = U_k @ D_k @ Z_k
    X = np.column_stack([X, T])

    X_full = X[map_idx[:, 1], :]

    # make UZ a blockdiagonal matrix with an M-dimensional identity matrix in its lower right block.
    # This way the full delta can be evaluated without having to split up gamma = [delta, alpha]
    UZ_full = np.zeros((UZ.shape[0] + M, k))
    UZ_full[: UZ.shape[0], : k - M] = UZ
    UZ_full[UZ.shape[0] :, k - M :] = np.eye(M)

    # create matrix W to rescale columns of X (see mgcv/src/tprs.c)
    # This step is not mentioned in the TP paper or the GAM-book
    # speeds up convergence immensely
    w = np.sqrt((X_full**2).sum(0) / n)
    W = np.diag(1 / w)
    X_full = X_full @ W
    S_full = W @ S_full @ W
    UZ_full = UZ_full @ W
    return X_full, S_full, UZ_full, map_idx


def get_FS(xk):
    """
    Create matrix F required to build the spline base and the penalizing matrix S,
    based on a set of knots xk (ascending order). Pretty much directly from p.201
    in Wood (2017).

    Parameters
    ----------
    xk : array-like
        Knots in strictly increasing order.

    Returns
    -------
    F : np.ndarray
    S : np.ndarray
    """
    xk = np.asarray(xk, dtype=np.float64).ravel()
    if xk.ndim != 1:
        raise ValueError("xk must be one-dimensional.")
    if xk.size < 3:
        raise ValueError("Need at least 3 knots.")
    if np.any(~np.isfinite(xk)):
        raise ValueError("Knots contain NaN or Inf.")
    if np.any(np.diff(xk) <= 0):
        raise ValueError("Knots must be strictly increasing.")

    k = len(xk)
    h = np.diff(xk)
    h_shift_up = h[1:]

    D = np.zeros((k - 2, k), dtype=np.float64)
    np.fill_diagonal(D, 1.0 / h[: k - 2])
    np.fill_diagonal(D[:, 1:], (-1.0 / h[: k - 2] - 1.0 / h_shift_up))
    np.fill_diagonal(D[:, 2:], 1.0 / h_shift_up)

    B = np.zeros((k - 2, k - 2), dtype=np.float64)
    np.fill_diagonal(B, (h[: k - 2] + h_shift_up) / 3.0)

    # Correct off-diagonals for irregular knot spacing.
    # The old version incorrectly used one scalar everywhere.
    np.fill_diagonal(B[:, 1:], h_shift_up / 6.0)
    np.fill_diagonal(B[1:, :], h_shift_up / 6.0)

    F_minus = np.linalg.solve(B, D)
    F = np.vstack([np.zeros(k, dtype=np.float64), F_minus, np.zeros(k, dtype=np.float64)])
    S = D.T @ F_minus
    return F, S


def cr_spl(x, n_knots, knots=None):
    """
    Build a cubic regression spline basis.

    Parameters
    ----------
    x : array-like
        Covariate values.
    n_knots : int
        Number of knots to generate if `knots` is None.
    knots : array-like or None
        Optional explicit knot locations. If supplied, these are used directly
        and `n_knots` is ignored.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if knots is None:
        n_knots = int(n_knots)
        if n_knots < 3:
            raise ValueError("At least 3 knots are required for cubic regression splines.")
        xu = np.unique(x)
        if xu.size < n_knots:
            raise ValueError(
                "Insufficient unique values to support the requested number of knots."
            )
        probs = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
        xk = np.quantile(xu, probs)
    else:
        xk = np.asarray(knots, dtype=np.float64).ravel()
        if xk.ndim != 1:
            raise ValueError("knots must be one-dimensional.")
        if xk.size < 3:
            raise ValueError("At least 3 knots are required for cubic regression splines.")
        if not np.all(np.isfinite(xk)):
            raise ValueError("knots contain NaN or Inf.")
        xk = np.unique(xk)
        if xk.size < 3:
            raise ValueError("Need at least 3 unique knots for cubic regression splines.")
        if np.any(np.diff(xk) <= 0):
            raise ValueError("knots must be strictly increasing.")

    n = len(x)
    k = len(xk)
    F, S = get_FS(xk)
    base = np.zeros((n, k), dtype=np.float64)

    for i in range(n):
        j = bisect.bisect_left(xk, x[i])
        if j == 0:
            j = 1
        if j >= len(xk):
            j = len(xk) - 1

        x_j = xk[j - 1]
        x_j1 = xk[j]
        h = x_j1 - x_j
        a_jm = (x_j1 - x[i]) / h
        a_jp = (x[i] - x_j) / h
        c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6.0
        c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6.0

        base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
        base[i, j - 1] += a_jm
        base[i, j] += a_jp

    return base, S, xk, F


def cr_spl_predict(x, knots, F):
    """
    Evaluate an existing cubic regression spline basis at new points.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    F = np.asarray(F, dtype=np.float64)

    n = len(x)
    k = len(knots)
    base = np.zeros((n, k), dtype=np.float64)

    for i in range(n):
        # Extrapolate outside knot range, following the original intended logic.
        if x[i] <= knots[0]:
            h = knots[1] - knots[0]
            xik = x[i] - knots[0]
            c_jm = -xik * h / 3.0
            c_jp = -xik * h / 6.0
            base[i, :] = c_jm * F[0, :] + c_jp * F[1, :]
            base[i, 0] += 1.0 - xik / h
            base[i, 1] += xik / h

        elif x[i] >= knots[-1]:
            j = len(knots) - 1
            h = knots[j] - knots[j - 1]
            xik = x[i] - knots[j]
            c_jm = xik * h / 6.0
            c_jp = xik * h / 3.0
            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += -xik / h
            base[i, j] += 1.0 + xik / h

        else:
            j = bisect.bisect_left(knots, x[i])
            if j == 0:
                j = 1

            x_j = knots[j - 1]
            x_j1 = knots[j]
            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6.0
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6.0

            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += a_jm
            base[i, j] += a_jp

    return base


def scale_penalty(basis, penalty):
    """
    Rescale the penalty matrix based on the design matrix of the smoother
    from mgcv to get penalties that react comparably to smoothing parameters.
    """
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    X_inf_norm = max(np.sum(np.abs(basis), axis=1)) ** 2
    S_norm = np.linalg.norm(penalty, ord=1)

    if X_inf_norm <= 0 or S_norm <= 0:
        return penalty.copy()

    norm = S_norm / X_inf_norm
    return penalty / norm


def identconst(basis, penalty):
    """
    Create constraint matrix and absorb identifiability constraint into model matrices:
    returns centered model matrices as well as orthogonal factor Z to map centered matrices
    back to unconstrained column space.
    """
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    constraint_matrix = basis.mean(axis=0).reshape(-1, 1)
    q, r = np.linalg.qr(constraint_matrix, mode="complete")
    penalty = np.double(
        np.linalg.multi_dot([np.transpose(q[:, 1:]), penalty, q[:, 1:]])
    )
    basis = basis @ q[:, 1:]
    return basis, penalty, q[:, 1:]


def pol2nb(pc):
    """
    Takes a dict of polygons and finds the neighbourhood-structure.
    Adapted from mgcv pol2nb.
    """
    num_poly = len(pc)
    lo1 = dict.fromkeys(pc.keys())
    hi1 = dict.fromkeys(pc.keys())
    lo2 = dict.fromkeys(pc.keys())
    hi2 = dict.fromkeys(pc.keys())

    for i in pc.keys():
        lo1[i] = min(pc[i][:, 0])
        lo2[i] = min(pc[i][:, 1])
        hi1[i] = max(pc[i][:, 0])
        hi2[i] = max(pc[i][:, 1])
        pc[i] = np.unique(pc[i], axis=0)

    ids = pc.keys()
    lo1 = list(lo1.values())
    lo2 = list(lo2.values())
    hi1 = list(hi1.values())
    hi2 = list(hi2.values())
    pc = list(pc.values())
    nb = dict.fromkeys(np.arange(0, num_poly))

    for k in range(num_poly):
        ol1 = np.logical_or(
            np.logical_or(
                np.logical_and(lo1[k] <= hi1, lo1[k] >= lo1),
                np.logical_and(hi1[k] <= hi1, hi1[k] >= lo1),
            ),
            np.logical_or(
                np.logical_and(lo1 <= hi1[k], lo1 >= lo1[k]),
                np.logical_and(hi1 <= hi1[k], hi1 >= lo1[k]),
            ),
        )
        ol2 = np.logical_or(
            np.logical_or(
                np.logical_and(lo2[k] <= hi2, lo2[k] >= lo2),
                np.logical_and(hi2[k] <= hi2, hi2[k] >= lo2),
            ),
            np.logical_or(
                np.logical_and(lo2 <= hi2[k], lo2 >= lo2[k]),
                np.logical_and(hi2 <= hi2[k], hi2 >= lo2[k]),
            ),
        )
        ol = np.logical_and(ol1, ol2)
        ol[k] = False
        ind = np.where(ol)[0]
        cok = pc[k]
        nb[k] = []

        if len(ind) > 0:
            for j in range(len(ind)):
                co = np.vstack([pc[ind[j]], cok])
                cou = np.unique(co, axis=0)
                n_shared = co.shape[0] - cou.shape[0]
                if n_shared > 0:
                    nb[k].append(ind[j])

    nb_mat = np.zeros((len(pc), len(pc)))
    for i in nb.keys():
        nb_mat[i, nb[i]] = -1
        nb_mat[i, i] = len(nb[i])

    nb_df = pd.DataFrame(nb_mat, columns=ids, index=ids)
    return nb_df


def mrf_design(regions, pc):
    """
    Function to create the design matrix for MRFSmooths. Simple indicator matrix.
    """
    regions = regions.astype("int")
    ids = pc.keys()
    design_mat = np.zeros([len(regions), len(ids)])
    design_df = pd.DataFrame(design_mat, columns=ids)
    for i in range(0, len(regions)):
        design_df.loc[i, regions[i]] = 1
    design_mat = design_df.to_numpy()
    return design_mat


def color_fader(c_1, c_2, mix=0):
    """
    Mix two colors as defined by mix in [0, 1].
    """
    c_1 = np.array(mpl.colors.to_rgb(c_1))
    c_2 = np.array(mpl.colors.to_rgb(c_2))
    if isinstance(mix, np.ndarray):
        cols = []
        for i in range(len(mix)):
            cols.append(mpl.colors.to_hex((1 - mix[i]) * c_1 + mix[i] * c_2))
        return cols
    else:
        return mpl.colors.to_hex((1 - mix) * c_1 + mix * c_2)


def color_bounds(values):
    """
    Helper for plotting MRFSmooths.
    """
    interval = np.linspace(0, 1, 100)
    min_v = min(values)[0]
    max_v = max(values)[0]
    mapped = min_v + ((max_v - min_v) / 1 - 0) * interval
    return mapped


def tp_T(data, M, m, d):
    """
    Get the polynomials of the features for which the penalty is null.
    """
    powers = poly_powers(m, d, M)
    n = data.shape[0]
    T = np.zeros((n, M))

    for i in range(M):
        T[:, i] = np.prod(data ** powers[i, :], axis=1)

    return T


def poly_powers(m, d, M):
    """
    Create an M x d matrix with the polynomial powers needed for model matrix T.
    One-to-one from mgcv/src/tprs.c: gen_tps_poly_powers
    """
    powers = np.zeros((M, d))
    index = np.zeros(d)

    for i in range(M):
        for j in range(d):
            powers[i, j] = index[j]

        s = 0
        for j in range(d):
            s += index[j]

        if s < (m - 1):
            index[0] += 1
        else:
            s -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                s += 1
                if s == m:
                    s -= index[j]
                    index[j] = 0
                else:
                    break

    return powers
