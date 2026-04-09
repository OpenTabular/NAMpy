import bisect

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
