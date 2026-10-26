"""Stable hurdle-Poisson likelihood kernels from mgcv/R/gamlss.r."""

import numpy as np
from scipy.special import gammaln

from ._scalar_math import cube, exp, log

_EPS = np.finfo(float).eps
_XMAX = np.finfo(float).max


def l1ee(x):
    """mgcv log(1-exp(-exp(x))), retaining its tail branches."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ex = exp(x)
        value = log(1 - exp(-ex))
        small = x < log(_EPS) / 3
        exi = ex[small]
        value[small] = log(exi - exi**2 / 2 + cube(exi) / 6)
        value[x < -log(_XMAX)] = x[x < -log(_XMAX)]
    return value


def lee1(x):
    """mgcv log(exp(exp(x))-1)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ex = exp(x)
        value = log(exp(ex) - 1)
        small = x < log(_EPS) / 3
        exi = ex[small]
        value[small] = log(exi + exi**2 / 2 + cube(exi) / 6)
        value[x < -log(_XMAX)] = x[x < -log(_XMAX)]
        large = x > log(log(_XMAX))
        value[large] = ex[large]
    return value


def ldg(g, deriv=4):
    """Derivatives of the positive-count log density in log Poisson mean."""
    g = np.atleast_1d(np.asarray(g, dtype=float))
    with np.errstate(all="ignore"):
        eg = exp(g)
        a = 1 + eg / 2 + eg**2 / 12
        ordinary = g > log(_EPS) / 3
        a[ordinary] = eg[ordinary] / (1 - exp(-eg[ordinary]))
        small = g < log(_EPS) / 3
        high = g > log(log(_XMAX)) + 1
        egi = eg[small]
        b = egi * (1 + egi / 6) / 2
        l2 = a * (a - eg - 1)
        l2[small] = a[small] * (b - egi)
        l2[high] = -eg[high]
        l3 = l4 = None
        if deriv > 1:
            l3 = a * (a * (-2 * a + 3 * (eg + 1)) - 3 * eg - eg**2 - 1)
            l3[small] = a[small] * (-b - 2 * b**2 + 3 * b * egi - egi**2)
            l3[high] = -eg[high]
        if deriv > 2:
            l4 = a * (
                6 * cube(a)
                - 12 * (eg + 1) * a**2
                + 4 * eg * a
                + 7 * (eg + 1) ** 2 * a
                - (4 + 3 * eg) * eg
                - cube(eg + 1)
            )
            l4[small] = a[small] * (
                6 * b * (3 + 3 * b + b**2)
                - 12 * egi * (1 + 2 * b + b**2)
                - 12 * b * (2 - b)
                + 4 * egi * (1 + b)
                + 7 * (egi**2 + 2 * egi + b * egi**2 + 2 * b * egi + b)
                - (4 + 3 * egi) * egi
                - egi * (3 + 3 * egi + egi**2)
            )
            l4[high] = -eg[high]
        l1 = -a
        limit = log(_XMAX) / 5
        capped = g > limit
        # R chained replacement also creates zero-filled higher derivative
        # arrays here when those arrays were previously NULL.
        for derivative in (l1, l2, l3, l4):
            if derivative is not None:
                derivative[capped] = -exp(limit)
        if np.any(capped):
            if l3 is None:
                l3 = np.zeros(g.size)
                l3[capped] = -exp(limit)
            if l4 is None:
                l4 = np.zeros(g.size)
                l4[capped] = -exp(limit)
    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


def lde(eta, deriv=4):
    """Derivatives of log probability of presence in cloglog predictor."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    with np.errstate(all="ignore"):
        small = eta < log(_EPS) / 3
        high = eta > log(_XMAX)
        et = exp(eta)
        eti = et[small]
        l1 = et.copy()
        l1[~small] = et[~small] / (exp(et[~small]) - 1)
        b = -eti * (1 + eti / 6) / 2
        l1[small] = 1 + b
        l1[high] = 0
        l2 = l1 * ((1 - et) - l1)
        l2[small] = -b * (1 + eti + b) - eti
        l2[high] = 0
        l3 = l4 = None
        if deriv > 1:
            high = eta > log(_XMAX) / 2
            l3 = l1 * ((1 - et) ** 2 - et - 3 * (1 - et) * l1 + 2 * l1**2)
            l3[small] = l1[small] * (
                -3 * eti + eti**2 - 3 * (-eti + b - eti * b) + 2 * b * (2 + b)
            )
            l3[high] = 0
        if deriv > 2:
            high = eta > log(_XMAX) / 3
            l4 = l1 * (
                (3 * et - 4) * et
                + 4 * et * l1
                + cube(1 - et)
                - 7 * (1 - et) ** 2 * l1
                + 12 * (1 - et) * l1**2
                - 6 * cube(l1)
            )
            l4[high] = 0
            l4[small] = l1[small] * (
                4 * l1[small] * eti
                - cube(eti)
                - b
                - 7 * b * eti**2
                - eti**2
                - 5 * eti
                - 10 * b * eti
                - 12 * eti * b**2
                - 6 * b**2
                - 6 * cube(b)
            )
    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


def zipll(y, g, eta, deriv=0):
    """Source log density and packed joint derivatives in (g,eta)."""
    y, g, eta = np.broadcast_arrays(
        np.atleast_1d(np.asarray(y, dtype=float)),
        np.atleast_1d(np.asarray(g, dtype=float)),
        np.atleast_1d(np.asarray(eta, dtype=float)),
    )
    zero = y == 0
    with np.errstate(all="ignore"):
        et = exp(eta)
        value = et.copy()
        value[zero] = -et[zero]
        value[~zero] = (
            l1ee(eta[~zero])
            + y[~zero] * g[~zero]
            - lee1(g[~zero])
            - gammaln(y[~zero] + 1)
        )
        p = 1 - exp(-et)
        l1 = l2 = l3 = l4 = expected = None
        if deriv:
            le, lg = lde(eta, deriv), ldg(g, deriv)
            l1 = np.zeros((y.size, 2))
            l2 = np.zeros((y.size, 3))
            expected = np.zeros_like(l2)
            l1[~zero, 0] = y[~zero] + lg["l1"][~zero]
            l1[zero, 1] = value[zero]
            l1[~zero, 1] = le["l1"][~zero]
            l2[~zero, 0] = lg["l2"][~zero]
            l2[~zero, 2] = le["l2"][~zero]
            l2[zero, 2] = value[zero]
            expected[:, 0] = p * lg["l2"]
            expected[:, 2] = -(1 - p) * et + p * le["l2"]
        if deriv > 1:
            l3 = np.zeros((y.size, 4))
            l3[~zero, 0] = lg["l3"][~zero]
            l3[~zero, 3] = le["l3"][~zero]
            l3[zero, 3] = value[zero]
        if deriv > 3:
            l4 = np.zeros((y.size, 5))
            l4[~zero, 0] = lg["l4"][~zero]
            l4[~zero, 4] = le["l4"][~zero]
            l4[zero, 4] = value[zero]
    return {"l": value, "l1": l1, "l2": l2, "l3": l3, "l4": l4, "El2": expected}
