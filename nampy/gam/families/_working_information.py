"""Observed-information derivatives for ordinary GLM family hooks."""

import numpy as np


def observed_weight_derivative(family, eta, y, order):
    """Chain the exponential-family deviance through its inverse link.

    Observed working information is D_eta_eta / 2. Differentiating the
    deviance rather than Fisher information retains its response dependence
    for noncanonical links, including binomial links.
    """
    mu = np.asarray(family.inverse_link(eta), dtype=float)
    residual = mu - np.asarray(y, dtype=float)
    v = np.asarray(family.variance(mu), dtype=float)
    v1, v2 = family.dvar(mu), family.d2var(mu)
    u = 1.0 / v
    u1 = -v1 / v**2
    u2 = 2.0 * v1**2 / v**3 - v2 / v**2
    d1 = 2.0 * residual * u
    d2 = 2.0 * u + 2.0 * residual * u1
    d3 = 4.0 * u1 + 2.0 * residual * u2
    a, b, c = (family.inverse_link_derivatives(eta, k) for k in (1, 2, 3))
    if order == 1:
        return (d3 * a**3 + 3.0 * d2 * a * b + d1 * c) / 2.0
    u3 = -6.0 * v1**3 / v**4 + 6.0 * v1 * v2 / v**3 - family.d3var(mu) / v**2
    d4 = 6.0 * u2 + 2.0 * residual * u3
    d = family.inverse_link_derivatives(eta, 4)
    return (
        d4 * a**4 + 6.0 * d3 * a**2 * b + 3.0 * d2 * b**2 + 4.0 * d2 * a * c + d1 * d
    ) / 2.0
