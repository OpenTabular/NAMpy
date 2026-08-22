import numpy as np
from scipy.linalg import eigh as scipy_eigh

from ...constraints.absorption import full_term_sum_to_zero_constraint
from ...penalties.algebra import scale_penalty
from ..basis.cr import cr_spl, cr_spl_predict


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


def add_full_rank_shrinkage(S, shrink=0.1, tol=1e-12, null_basis=None, knots=None):
    """
    Make a symmetric penalty full rank by shrinking its null space.

    Mirror mgcv/R/smooth.r::smooth.construct.cr.smooth.spec(): eigen-decompose
    the explicitly symmetrized raw CR penalty, then replace the trailing zero
    eigenvalues by small positive multiples of the smallest positive eigenvalue.
    """
    del tol, knots, null_basis
    penalty = np.asarray(S, dtype=np.float64)
    penalty = 0.5 * (penalty + penalty.T)
    # Mirror smooth.construct.cr.smooth.spec() through SciPy's public symmetric
    # eigensolver API. The nominally-null eigenvectors remain platform-dependent
    # upstream too, so their exact orientation is not a parity contract.
    values, vectors = scipy_eigh(
        penalty,
        lower=True,
        check_finite=False,
    )
    values = np.asarray(values[::-1], dtype=np.float64)
    vectors = np.asarray(vectors[:, ::-1], dtype=np.float64)
    nk = int(values.size)
    if nk < 3:
        raise ValueError("cubic regression spline penalty requires k >= 3.")

    # Operation-for-operation port of smooth.construct.cr.smooth.spec(): for
    # a cs smooth, smooth.construct.cs.smooth.spec() sets shrink=.1 before
    # delegating here, and the final two eigenvalues are replaced in order.
    values[nk - 2] = values[nk - 3] * float(shrink)
    values[nk - 1] = values[nk - 2] * float(shrink)
    return np.asarray(vectors @ (values[:, None] * vectors.T), dtype=np.float64)


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


class CubicSplines:
    """
    Cubic regression spline basis with both raw and constrained representations.
    """

    def __init__(self, x, k, knots=None):
        X_raw, S_raw_unscaled, knots, F = cr_spl(x, n_knots=k, knots=knots)

        # Store the penalty before scale_penalty — needed by te/ti for eigenvalue
        # normalization of marginal penalties before building the tensor product.
        self.raw_penalty_unscaled = np.asarray(S_raw_unscaled, dtype=np.float64)

        S_raw = scale_penalty(X_raw, S_raw_unscaled)
        # Match mgcv absorb.cons centering through GAM constraint absorption policy.
        X_centered, penalties_centered, center_mat = full_term_sum_to_zero_constraint(
            X_raw, [S_raw]
        )
        S_centered = penalties_centered[0]

        self.raw_basis = X_raw
        self.raw_penalty = S_raw

        self.basis = X_centered
        self.penalty = S_centered
        self.center_mat = center_mat

        self.knots = knots
        self.dim_basis = X_centered.shape[1]
        self.F = F

        # cr smooths set noterp=TRUE in mgcv, meaning the np=TRUE reparameterization
        # is skipped for cr marginals in te/ti tensor products.  The XP transforms
        # are therefore never applied to cr spline marginals.
        self._np_transform = None
        self._np_transform_centered = None

    def transform_new_raw(self, x_new):
        return cr_spl_predict(x_new, knots=self.knots, F=self.F)

    def transform_new_centered(self, x_new):
        return self.transform_new_raw(x_new) @ self.center_mat

    def transform_new(self, x_new):
        return self.transform_new_raw(x_new)
