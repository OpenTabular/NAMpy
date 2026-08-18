import numpy as np
from scipy.linalg import solveh_banded


def _r_quantile_type7_sorted(values, probs):
    """Evaluate R's default type-7 quantile on sorted values."""
    values = np.asarray(values, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()

    # Mirror stats::quantile.default(type = 7).  The weighted expression is
    # deliberately kept in R's operand order: np.quantile's alternative lerp
    # ordering can move an automatic CR knot by one ULP, which changes mgcv's
    # numerically selected cs null-space eigenvectors.
    index = 1.0 + (values.size - 1) * probs
    lo = np.floor(index).astype(np.intp)
    hi = np.ceil(index).astype(np.intp)
    quantiles = values[lo - 1].copy()
    interpolate = (index > lo) & (values[hi - 1] != quantiles)
    h = index[interpolate] - lo[interpolate]
    quantiles[interpolate] = (
        (1.0 - h) * quantiles[interpolate] + h * values[hi[interpolate] - 1]
    )
    return quantiles


def get_FS(xk):
    """
    Create matrix F required to build spline basis and penalizing matrix S.
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
    n2 = k - 2

    # Port mgcv/src/mgcv.c::getFS: build D as an n2 x k RHS, solve the
    # symmetric tridiagonal B system, then assemble D'B^{-1}D with the same
    # row loops.
    D = np.zeros((n2, k), dtype=np.float64, order="F")
    for i in range(n2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 2] = 1.0 / h[i + 1]
        D[i, i + 1] = -D[i, i] - D[i, i + 2]

    ldB = np.asarray((h[:n2] + h[1:]) / 3.0, dtype=np.float64)
    sdB = np.asarray(h[1:n2] / 6.0, dtype=np.float64)
    banded_B = np.zeros((2, n2), dtype=np.float64)
    banded_B[0, 1:] = sdB
    banded_B[1, :] = ldB
    F_minus = solveh_banded(
        banded_B,
        D,
        overwrite_b=False,
        lower=False,
        check_finite=False,
    )

    F = np.vstack(
        [np.zeros(k, dtype=np.float64), F_minus, np.zeros(k, dtype=np.float64)]
    )

    S = np.zeros((k, k), dtype=np.float64)
    a = 1.0 / h[0]
    S[0, :] = F_minus[0, :] * a
    if k > 3:
        a = -1.0 / h[0] - 1.0 / h[1]
        b = 1.0 / h[1]
        S[1, :] = F_minus[0, :] * a + F_minus[1, :] * b
        for j in range(2, n2):
            a = 1.0 / h[j - 1]
            c = 1.0 / h[j]
            b = -a - c
            S[j, :] = F_minus[j - 2, :] * a + F_minus[j - 1, :] * b + F_minus[j, :] * c
        j = n2
        a = 1.0 / h[j - 1]
        b = -1.0 / h[j - 1] - 1.0 / h[j]
        S[n2, :] = F_minus[n2 - 2, :] * a + F_minus[n2 - 1, :] * b
    else:
        a = -1.0 / h[0] - 1.0 / h[1]
        S[1, :] = F_minus[0, :] * a
    j = n2
    a = 1.0 / h[j]
    S[k - 1, :] = F_minus[n2 - 1, :] * a
    return F, S


def cr_spl(x, n_knots, knots=None):
    """
    Build a cubic regression spline basis.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if knots is None:
        n_knots = int(n_knots)
        if n_knots < 3:
            raise ValueError(
                "At least 3 knots are required for cubic regression splines."
            )
        xu = np.unique(x)
        if xu.size < n_knots:
            raise ValueError(
                "Insufficient unique values to support the requested number of knots."
            )
        probs = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
        xk = _r_quantile_type7_sorted(xu, probs)
    else:
        xk = np.asarray(knots, dtype=np.float64).ravel()
        if xk.ndim != 1:
            raise ValueError("knots must be one-dimensional.")
        if xk.size < 3:
            raise ValueError(
                "At least 3 knots are required for cubic regression splines."
            )
        if not np.all(np.isfinite(xk)):
            raise ValueError("knots contain NaN or Inf.")
        xk = np.unique(xk)
        if xk.size < 3:
            raise ValueError(
                "Need at least 3 unique knots for cubic regression splines."
            )
        if np.any(np.diff(xk) <= 0):
            raise ValueError("knots must be strictly increasing.")

    n = len(x)
    k = len(xk)
    F, S = get_FS(xk)
    base = np.zeros((n, k), dtype=np.float64)

    j = np.searchsorted(xk, x, side="left")
    j = np.clip(j, 1, k - 1)

    x_j = xk[j - 1]
    x_j1 = xk[j]
    h = x_j1 - x_j
    left = x_j1 - x
    right = x - x_j
    a_jm = left / h
    a_jp = right / h
    c_jm = (left**3 / h - h * left) / 6.0
    c_jp = (right**3 / h - h * right) / 6.0

    base[:, :] = c_jm[:, None] * F[j - 1, :] + c_jp[:, None] * F[j, :]
    rows = np.arange(n)
    base[rows, j - 1] += a_jm
    base[rows, j] += a_jp

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

    left_mask = x <= knots[0]
    if np.any(left_mask):
        h = knots[1] - knots[0]
        xik = x[left_mask] - knots[0]
        c_jm = -xik * h / 3.0
        c_jp = -xik * h / 6.0
        base[left_mask, :] = c_jm[:, None] * F[0, :] + c_jp[:, None] * F[1, :]
        base[left_mask, 0] += 1.0 - xik / h
        base[left_mask, 1] += xik / h

    right_mask = x >= knots[-1]
    if np.any(right_mask):
        j = k - 1
        h = knots[j] - knots[j - 1]
        xik = x[right_mask] - knots[j]
        c_jm = xik * h / 6.0
        c_jp = xik * h / 3.0
        base[right_mask, :] = c_jm[:, None] * F[j - 1, :] + c_jp[:, None] * F[j, :]
        base[right_mask, j - 1] += -xik / h
        base[right_mask, j] += 1.0 + xik / h

    interior_mask = ~(left_mask | right_mask)
    if np.any(interior_mask):
        x_mid = x[interior_mask]
        j = np.searchsorted(knots, x_mid, side="left")
        j = np.clip(j, 1, k - 1)

        x_j = knots[j - 1]
        x_j1 = knots[j]
        h = x_j1 - x_j
        left = x_j1 - x_mid
        right = x_mid - x_j
        a_jm = left / h
        a_jp = right / h
        c_jm = (left**3 / h - h * left) / 6.0
        c_jp = (right**3 / h - h * right) / 6.0

        base[interior_mask, :] = c_jm[:, None] * F[j - 1, :] + c_jp[:, None] * F[j, :]
        rows = np.flatnonzero(interior_mask)
        base[rows, j - 1] += a_jm
        base[rows, j] += a_jp

    return base


def cr_exact_null_basis_from_knots(knots):
    """
    Return the 2-column orthonormal null-space basis of the cubic regression
    spline penalty from knot positions.  Used by tensor reparameterization.
    """
    knots = np.asarray(knots, dtype=np.float64).ravel()
    u1 = knots - knots[0]
    u1 = u1 / float(np.linalg.norm(u1))
    one = np.ones_like(knots)
    u2 = one - u1 * float(u1 @ one)
    u2 = u2 / float(np.linalg.norm(u2))
    return np.column_stack([u1, u2])
