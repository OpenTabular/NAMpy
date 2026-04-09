# splines/univariate_bases.py
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh as _scipy_eigh


def place_knots_through_values(x, nk):
    """
    mgcv::place.knots analogue for 1D cyclic cubic setup.

    Places nk knots evenly through the ordered unique covariate values,
    with the first and last knots at the data extremes.
    """
    x = np.sort(np.unique(np.asarray(x, dtype=np.float64).ravel()))
    n = len(x)

    if nk > n:
        raise ValueError("more knots than unique data values is not allowed")
    if nk < 2:
        raise ValueError("too few knots")
    if nk == 2:
        return np.array([x[0], x[-1]], dtype=np.float64)

    delta = (n - 1) / float(nk - 1)
    idx = np.arange(1, nk - 1, dtype=np.float64)
    lbi = np.floor(delta * idx + 1).astype(int)
    frac = delta * idx + 1 - lbi

    x_shift = x[1:]
    knot = np.zeros(nk, dtype=np.float64)
    knot[0] = x[0]
    knot[-1] = x[-1]
    knot[1:-1] = x[lbi - 1] * (1.0 - frac) + x_shift[lbi - 1] * frac
    return knot


def add_full_rank_shrinkage(S, shrink=0.1, tol=1e-12):
    """
    Make a symmetric penalty full rank by shrinking its null space.

    This mirrors mgcv's cs/ts construction idea: null-space eigenvalues are
    replaced by small positive multiples of the smallest positive eigenvalue.
    """
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)

    # scipy.linalg.eigh (DSYEVR) matches R's eigen(symmetric=TRUE) null-space
    # eigenvectors to machine precision.  np.linalg.eigh (DSYEVD) finds a
    # different rotation of the 2-D null space, producing a ~1e-2 penalty
    # discrepancy for cs after assigning different shrinkage values to the two
    # null directions.
    evals, U = _scipy_eigh(S)
    tol_eff = tol * max(1.0, np.max(np.abs(evals)) if evals.size else 1.0)

    pos = evals[evals > tol_eff]
    if pos.size == 0:
        return S.copy()

    out = evals.copy()
    null_idx = np.where(evals <= tol_eff)[0]

    if null_idx.size:
        base = float(np.min(pos))
        # keep the null-space penalties ordered and smaller than the range-space penalties
        for j, idx in enumerate(null_idx):
            out[idx] = base * (shrink ** (null_idx.size - j))

    return (U * out) @ U.T


def cyclic_wrap(x0, x1, x):
    """
    Wrap x onto [x0, x1] for periodic prediction.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    h = float(x1 - x0)
    if h <= 0:
        raise ValueError("cyclic interval must have positive width.")

    if np.max(x) > x1:
        ind = x > x1
        x[ind] = x0 + np.mod(x[ind] - x1, h)

    if np.min(x) < x0:
        ind = x < x0
        x[ind] = x1 - np.mod(x0 - x[ind], h)

    return x


def cyclic_cubic_bd(knots):
    """
    Build the BD matrix used by mgcv's cyclic cubic basis.

    If p are spline values at the knots (excluding the repeated endpoint),
    and m are the second derivatives at those knots, then:
        m = BD @ p
    """
    knots = np.asarray(knots, dtype=np.float64).ravel()

    if knots.ndim != 1 or knots.size < 4:
        raise ValueError("cyclic cubic splines require at least 4 knots.")
    if np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be strictly increasing.")

    h = knots[1:] - knots[:-1]
    n = knots.size - 1

    B = np.zeros((n, n), dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)

    B[0, 0] = (h[n - 1] + h[0]) / 3.0
    B[0, 1] = h[0] / 6.0
    B[0, n - 1] = h[n - 1] / 6.0
    D[0, 0] = -(1.0 / h[0] + 1.0 / h[n - 1])
    D[0, 1] = 1.0 / h[0]
    D[0, n - 1] = 1.0 / h[n - 1]

    for i in range(1, n - 1):
        B[i, i - 1] = h[i - 1] / 6.0
        B[i, i] = (h[i - 1] + h[i]) / 3.0
        B[i, i + 1] = h[i] / 6.0

        D[i, i - 1] = 1.0 / h[i - 1]
        D[i, i] = -(1.0 / h[i - 1] + 1.0 / h[i])
        D[i, i + 1] = 1.0 / h[i]

    B[n - 1, n - 2] = h[n - 2] / 6.0
    B[n - 1, n - 1] = (h[n - 2] + h[n - 1]) / 3.0
    B[n - 1, 0] = h[n - 1] / 6.0
    D[n - 1, n - 2] = 1.0 / h[n - 2]
    D[n - 1, n - 1] = -(1.0 / h[n - 2] + 1.0 / h[n - 1])
    D[n - 1, 0] = 1.0 / h[n - 1]

    BD = np.linalg.solve(B, D)
    return BD, B, D


def cyclic_cubic_predict_matrix(x, knots, BD):
    """
    mgcv::Predict.matrix.cyclic.smooth analogue in pure Python.

    knots has length nk; basis width is nk - 1 because the endpoint is repeated
    cyclically.
    """
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    knots = np.asarray(knots, dtype=np.float64).ravel()

    n = knots.size
    if n < 4:
        raise ValueError("cyclic cubic splines require at least 4 knots.")

    h = knots[1:] - knots[:-1]

    if np.max(x) > np.max(knots) or np.min(x) < np.min(knots):
        x = cyclic_wrap(np.min(knots), np.max(knots), x)

    j = x.copy()
    for i in range(n, 1, -1):
        j[x <= knots[i - 1]] = i
    j = j.astype(int)

    j1 = j - 1
    hj = j - 1
    j = j.copy()
    j[j == n] = 1

    eye = np.eye(n - 1, dtype=np.float64)

    X = (
        BD[j1 - 1, :] * ((knots[j1] - x)[:, None] ** 3) / (6.0 * h[hj - 1])[:, None]
        + BD[j - 1, :]
        * ((x - knots[j1 - 1])[:, None] ** 3)
        / (6.0 * h[hj - 1])[:, None]
        - BD[j1 - 1, :] * (h[hj - 1] * (knots[j1] - x) / 6.0)[:, None]
        - BD[j - 1, :] * (h[hj - 1] * (x - knots[j1 - 1]) / 6.0)[:, None]
        + eye[j1 - 1, :] * ((knots[j1] - x) / h[hj - 1])[:, None]
        + eye[j - 1, :] * ((x - knots[j1 - 1]) / h[hj - 1])[:, None]
    )
    return np.asarray(X, dtype=np.float64)


def bspline_design_matrix(x, knots, degree, deriv=0, extrapolate=True):
    """
    Dense B-spline design matrix using scipy.interpolate.BSpline.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()

    degree = int(degree)
    deriv = int(deriv)

    n_basis = knots.size - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector / degree combination.")

    if deriv > degree:
        return np.zeros((x.size, n_basis), dtype=np.float64)

    X = np.empty((x.size, n_basis), dtype=np.float64)
    for i in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[i] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=extrapolate)
        if deriv > 0:
            spl = spl.derivative(deriv)
        X[:, i] = spl(x)
    return X


def pspline_knots(x, bs_dim, basis_order, supplied_knots=None):
    """
    mgcv::smooth.construct.ps.smooth.spec-style knot setup.

    basis_order = m[0], penalty order handled separately.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    m1 = int(basis_order)

    nk = int(bs_dim) - m1
    if nk <= 0:
        raise ValueError("basis dimension too small for b-spline order")
    if supplied_knots is None and nk < 2:
        raise ValueError(
            "Automatic P-spline knot construction requires bs.dim > basis_order + 1."
        )

    if supplied_knots is None:
        xl = float(np.min(x))
        xu = float(np.max(x))
        xr = xu - xl
        xl = xl - xr * 0.001
        xu = xu + xr * 0.001
        dx = (xu - xl) / float(nk - 1)
        return np.linspace(
            xl - dx * (m1 + 1),
            xu + dx * (m1 + 1),
            nk + 2 * m1 + 2,
        )

    k = np.asarray(supplied_knots, dtype=np.float64).ravel()
    if k.size == 2:
        xl = float(np.min(k))
        xu = float(np.max(k))
        if xl > np.min(x) or xu < np.max(x):
            raise ValueError("knot range does not include data")
        xr = xu - xl
        xl = xl - xr * 0.001
        xu = xu + xr * 0.001
        if nk < 2:
            raise ValueError(
                "Automatic P-spline knot construction requires bs.dim > basis_order + 1."
            )
        dx = (xu - xl) / float(nk - 1)
        return np.linspace(
            xl - dx * (m1 + 1),
            xu + dx * (m1 + 1),
            nk + 2 * m1 + 2,
        )

    expected = nk + 2 * m1 + 2
    if k.size != expected:
        raise ValueError(f"there should be {expected} supplied knots")
    return k


def pspline_difference_penalty(n_coef, diff_order):
    """
    Difference penalty for a P-spline basis.
    """
    n_coef = int(n_coef)
    diff_order = int(diff_order)

    if diff_order < 0:
        raise ValueError("diff_order must be >= 0")
    if diff_order > n_coef - 1:
        raise ValueError("penalty order too high for basis dimension")

    D = np.eye(n_coef, dtype=np.float64)
    if diff_order > 0:
        D = np.diff(D, n=diff_order, axis=0)
    return D.T @ D


def pspline_predict_matrix(x, knots, basis_order, deriv=0):
    """
    mgcv::Predict.matrix.pspline.smooth analogue.

    Uses ordinary B-spline evaluation inside the inner knot range and
    linear extrapolation of the smooth outside that range.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    deriv = int(deriv)

    degree = int(basis_order) + 1
    n_basis = knots.size - degree - 1

    if deriv > degree:
        return np.zeros((x.size, n_basis), dtype=np.float64)

    ll = knots[int(basis_order) + 1]
    ul = knots[len(knots) - int(basis_order) - 2]

    X = np.zeros((x.size, n_basis), dtype=np.float64)
    inside = (x >= ll) & (x <= ul)

    if np.any(inside):
        X[inside, :] = bspline_design_matrix(
            x[inside],
            knots,
            degree=degree,
            deriv=deriv,
            extrapolate=True,
        )

    if np.all(inside):
        return X

    if deriv >= 2:
        return X

    B_ll = bspline_design_matrix(
        np.array([ll], dtype=np.float64),
        knots,
        degree=degree,
        deriv=0,
        extrapolate=True,
    )[0]
    dB_ll = bspline_design_matrix(
        np.array([ll], dtype=np.float64),
        knots,
        degree=degree,
        deriv=1,
        extrapolate=True,
    )[0]
    B_ul = bspline_design_matrix(
        np.array([ul], dtype=np.float64),
        knots,
        degree=degree,
        deriv=0,
        extrapolate=True,
    )[0]
    dB_ul = bspline_design_matrix(
        np.array([ul], dtype=np.float64),
        knots,
        degree=degree,
        deriv=1,
        extrapolate=True,
    )[0]

    left = x < ll
    if np.any(left):
        if deriv == 0:
            X[left, :] = B_ll[None, :] + (x[left] - ll)[:, None] * dB_ll[None, :]
        else:
            X[left, :] = np.broadcast_to(dB_ll, (np.sum(left), n_basis))

    right = x > ul
    if np.any(right):
        if deriv == 0:
            X[right, :] = B_ul[None, :] + (x[right] - ul)[:, None] * dB_ul[None, :]
        else:
            X[right, :] = np.broadcast_to(dB_ul, (np.sum(right), n_basis))

    return X
