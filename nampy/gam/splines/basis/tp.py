import math

import numpy as np


def eta(E, m, d):
    E = np.asarray(E, dtype=np.float64)

    if d % 2 == 0:
        d_half = d // 2
        const = ((-1) ** (m + 1 + d_half)) / (
            2 ** (2 * m - 1)
            * np.pi**d_half
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
