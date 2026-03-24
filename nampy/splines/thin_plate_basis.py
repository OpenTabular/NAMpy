import math

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance_matrix


def eta(E, m, d):
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


def poly_powers(m, d, M):
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


def tp_T(data, M, m, d):
    powers = poly_powers(m, d, M)
    n = data.shape[0]
    T = np.zeros((n, M))

    for i in range(M):
        T[:, i] = np.prod(data ** powers[i, :], axis=1)

    return T


def tp_spline(x, k, pen_order, n, d, M):
    if d == 1:
        x = x - x.mean()

    x_un = np.unique(x, axis=0)
    map_idx = np.all(
        (np.expand_dims(np.array(x_un), 0) == np.expand_dims(x, 1)), axis=2
    )
    map_idx = np.argwhere(map_idx)

    E = distance_matrix(x_un, x_un)
    E = eta(E, pen_order, d)

    eigen_values, U = eigsh(E, k, which="LA")
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    U = U[:, idx]
    D = np.diag(eigen_values)

    U_k = U[:, :k]
    D_k = D[:k, :k]
    T = tp_T(x_un, M, pen_order, d)

    q, _ = np.linalg.qr(np.dot(U_k.T, T), mode="complete")
    Z_k = q[:, M:]

    UZ = U_k @ Z_k
    S = Z_k.T @ D_k @ Z_k
    S_full = np.zeros((k, k))
    S_full[: k - M, : k - M] = S

    X = U_k @ D_k @ Z_k
    X = np.column_stack([X, T])
    X_full = X[map_idx[:, 1], :]

    UZ_full = np.zeros((UZ.shape[0] + M, k))
    UZ_full[: UZ.shape[0], : k - M] = UZ
    UZ_full[UZ.shape[0] :, k - M :] = np.eye(M)

    w = np.sqrt((X_full**2).sum(0) / n)
    W = np.diag(1 / w)
    X_full = X_full @ W
    S_full = W @ S_full @ W
    UZ_full = UZ_full @ W
    return X_full, S_full, UZ_full, map_idx
