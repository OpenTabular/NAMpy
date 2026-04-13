import numpy as np


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
    h_shift_up = h[1:]

    D = np.zeros((k - 2, k), dtype=np.float64)
    np.fill_diagonal(D, 1.0 / h[: k - 2])
    np.fill_diagonal(D[:, 1:], (-1.0 / h[: k - 2] - 1.0 / h_shift_up))
    np.fill_diagonal(D[:, 2:], 1.0 / h_shift_up)

    B = np.zeros((k - 2, k - 2), dtype=np.float64)
    np.fill_diagonal(B, (h[: k - 2] + h_shift_up) / 3.0)
    np.fill_diagonal(B[:, 1:], h_shift_up / 6.0)
    np.fill_diagonal(B[1:, :], h_shift_up / 6.0)

    F_minus = np.linalg.solve(B, D)
    F = np.vstack(
        [np.zeros(k, dtype=np.float64), F_minus, np.zeros(k, dtype=np.float64)]
    )
    S = D.T @ F_minus
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
        xk = np.quantile(xu, probs)
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

        base[interior_mask, :] = (
            c_jm[:, None] * F[j - 1, :] + c_jp[:, None] * F[j, :]
        )
        rows = np.flatnonzero(interior_mask)
        base[rows, j - 1] += a_jm
        base[rows, j] += a_jp

    return base


def cr_exact_null_basis_from_knots(knots):
    """
    Return the 2-column orthonormal null-space basis of the cubic regression
    spline penalty from knot positions.  Used by t2 tensor reparameterization.
    """
    knots = np.asarray(knots, dtype=np.float64).ravel()
    u1 = knots - knots[0]
    u1 = u1 / float(np.linalg.norm(u1))
    one = np.ones_like(knots)
    u2 = one - u1 * float(u1 @ one)
    u2 = u2 / float(np.linalg.norm(u2))
    return np.column_stack([u1, u2])
