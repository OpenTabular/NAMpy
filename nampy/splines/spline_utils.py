import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
import bisect
import matplotlib as mpl


def eta(E, m, d):
    """
    Calculate the eta function given a matrix of Euclidean distances, penalty order and dimensionality
    of the data
    :param E: Matrix of euclidean distances between points
    :param m: penalty order
    :param d: dimensionality of data
    :return: eta-fct of the supplied Euclidean distances
    """
    if d % 2 == 0:
        const = ((-1) ** (m + 1 + d / 2)) / (
            2 ** (2 * m - 1)
            * np.pi ** (d / 2)
            * np.math.factorial(m - 1)
            * np.math.factorial(m - d / 2)
        )
        E = const * E ** (2 * m - d) * np.log(E)

    else:
        E = (
            np.math.gamma(d / 2 - m)
            / (2 ** (2 * m) * np.pi ** (d / 2) * np.math.factorial(m - 1))
            * E ** (2 * m - d)
        )
    np.nan_to_num(E, 0)
    return E


def tp_spline(x, k, pen_order, n, d, M):

    # subtract mean from data (try to recreate model matrix in mgcv, did not work. Doesn't change the model, so can be
    # ignored
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
    based on a set of knots xk (ascending order). Pretty much directly from p.201 in Wood (2017)
    :param xk: knots (for now always np.linspace(x.min(), x.max(), n_knots)
    """
    k = len(xk)
    h = np.diff(xk)
    h_shift_up = h.copy()[1:]

    D = np.zeros((k - 2, k))
    np.fill_diagonal(D, 1 / h[: k - 2])
    np.fill_diagonal(D[:, 1:], (-1 / h[: k - 2] - 1 / h_shift_up))
    np.fill_diagonal(D[:, 2:], 1 / h_shift_up)

    B = np.zeros((k - 2, k - 2))
    np.fill_diagonal(B, (h[: k - 2] + h_shift_up) / 3)
    np.fill_diagonal(B[:, 1:], h_shift_up[k - 3] / 6)
    np.fill_diagonal(B[1:, :], h_shift_up[k - 3] / 6)
    F_minus = np.linalg.inv(B) @ D
    F = np.vstack([np.zeros(k), F_minus, np.zeros(k)])
    S = D.T @ np.linalg.inv(B) @ D
    return F, S


def cr_spl(x, n_knots):
    """

    :param x: x values to be evalutated
    :param n_knots: number of knots
    :return:
    """

    xk = np.linspace(x.min(), x.max(), n_knots)
    n = len(x)
    k = len(xk)
    F, S = get_FS(xk)
    base = np.zeros((n, k))
    for i in range(0, len(x)):
        # find interval in which x[i] lies
        # and evaluate basis function from p.201 in Wood (2017)
        j = bisect.bisect_left(xk, x[i])
        if j == 0:
            j = 1
        x_j = xk[j - 1]
        x_j1 = xk[j]
        h = x_j1 - x_j
        a_jm = (x_j1 - x[i]) / h
        a_jp = (x[i] - x_j) / h
        c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
        c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
        base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
        base[i, j - 1] += a_jm
        base[i, j] += a_jp
    return base, S, xk, F


def cr_spl_predict(x, knots, F):
    """
    pretty much the same as cr_spl, this time evaluating it for already given knots and F
    (could probably just be integrated into cr_spl)
    """

    n = len(x)
    k = len(knots)
    base = np.zeros((n, k))
    for i in range(0, len(x)):
        # in case x[i] is lies outside the range of the knots, extrapolate (see mgcv/src/mgcv.c (crspl) on github)
        if x[i] < min(knots):
            j = 0
            h = knots[1] - knots[0]
            xik = x[i] - knots[0]
            c_jm = -xik * h / 3
            c_jp = -xik * h / 6
            base[i, :] = c_jm * F[0, :] + c_jp * F[1, :]
            base[i, 0] += 1 - xik / h
            base[i, 1] += xik / h
        elif x[i] > max(knots):
            j = len(knots) - 1
            h = knots[j] - knots[j - 1]
            xik = x[i] - knots[j]
            c_jm = xik * h / 6
            c_jp = xik * h / 3
            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += -xik / h
            base[i, j] += 1 + xik / h
        # find interval in which x[i] lies and evaluate accordingly just like in cr_spl
        else:
            j = bisect.bisect_left(knots, x[i])
            if j == 0:
                j = 1
            x_j = knots[j - 1]
            x_j1 = knots[j]
            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += a_jm
            base[i, j] += a_jp
    return base


def scale_penalty(basis, penalty):
    """
    rescale the penalty matrix based on the design matrix of the smoother
    from mgcv to get penalties that react comparably to smoothing parameters
    (works for CubicSplines and MRFSmooth, not for TPSpline since the model
    matrices are completely different)
    """
    X_inf_norm = max(np.sum(abs(basis), axis=1)) ** 2
    S_norm = np.linalg.norm(penalty, ord=1)
    norm = S_norm / X_inf_norm
    penalty = penalty / norm
    return penalty


def identconst(basis, penalty):
    """
    create constraint matrix and absorb identifiability constraint into model matrices:
    returns centered model matrices as well orthogonal factor Z to map centered matrices
    back to unconstrained column space
    """
    constraint_matrix = basis.mean(axis=0).reshape(-1, 1)
    q, r = np.linalg.qr(constraint_matrix, mode="complete")
    penalty = np.double(
        np.linalg.multi_dot([np.transpose(q[:, 1:]), penalty, q[:, 1:]])
    )
    basis = basis @ q[:, 1:]
    return basis, penalty, q[:, 1:]


def pol2nb(pc):
    """
    Takes a dict of polygons and finds the neighbourhood-structure. (Works by finding possible neighbour candidates
    -> if points are shared, the polygons are neighbours. Function adapted from mgcv pol2nb). The neighbourhood-structure
    functions as the penalty matrix for MRFSmooth.
    :param pc: dict of polygons
    :return: neighbourhood-structure as pd.DataFrame (so I could have named cols and rows)
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
    :param regions: x
    :param pc: dict of polygons
    :return: design matrix with columns in order in which they are in the dictionary of polygons
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
    Function that takes to colors as inputs and mixes them as defined by mix [0, 1]. If the input is an array, function
    returns a list of color codes, else just a single color code. (Necessary for MRFSmooth plot method)
    :param c_1: color 1
    :param c_2: color 2
    :param mix: value between 0 and 1
    :return:
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
    Also function for plotting MRFSmooths: Finds max and min of provided values and creates and maps the interval
    between the two on the interval 0 to 1. Allows me to create a colorbar with the correct ticks.
    :param values: estimated parameters of MRFSmooth
    :return: m
    """
    interval = np.linspace(0, 1, 100)
    min_v = min(values)[0]
    max_v = max(values)[0]
    mapped = min_v + ((max_v - min_v) / 1 - 0) * interval
    return mapped


def tp_T(data, M, m, d):
    """
    function to get the polynomials of the features for which the penalty is null.
    Currently calls a c-function from mgcv which returns the polynomial powers of
    the M functions. The returned values are than used to transform the data
    :param data: data
    :param M: size of nullspace
    :param m: penalty order
    :param d: dimensions of data
    :return:
    """

    # call poly_powers to get the polynomial powers with which to evaluate x
    powers = poly_powers(m, d, M)
    n = data.shape[0]
    T = np.zeros((n, M))

    # loop through row of powers
    for i in range(M):
        T[:, i] = np.prod(data ** powers[i, :], axis=1)

    return T


def poly_powers(m, d, M):
    """
    one to one translation from a function in mgcv that creates an M x d matrix
    with the polynomial powers needed for model matrix T
    Parameters
    ----------
    m: penalty order
    d: dimensions
    M: nullspace dim

    Returns matrix of polynomial powers at which to evaluate data.
    One to one from mgcv (https://github.com/cran/mgcv/blob/master/src/tprs.c: gen_tps_poly_powers)
    -------

    """

    powers = np.zeros((M, d))
    index = np.zeros(d)
    for i in range(M):
        for j in range(d):
            powers[i, j] = index[j]
        sum = 0
        for j in range(d):
            sum += index[j]
        if sum < (m - 1):
            index[0] += 1
        else:
            sum -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                sum += 1
                if sum == m:
                    sum -= index[j]
                    index[j] = 0
                else:
                    break
    return powers
