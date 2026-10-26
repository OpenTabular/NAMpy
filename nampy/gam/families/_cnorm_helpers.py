"""Signed normal-tail arithmetic from mgcv/R/misc.r and efam.r::cnorm."""

import warnings

import numpy as np

from ._scalar_math import exp, expm1, log, log1p

_LOG_2PI = log(2 * np.pi)
_CUTOFF = log(1 / np.finfo(float).eps) + 1


def logexm1(x):
    x = np.asarray(x, dtype=float).copy()
    if np.any(np.isnan(x)):
        raise ValueError("NAs are not allowed in subscripted assignments")
    small = x < _CUTOFF
    with np.errstate(divide="ignore", invalid="ignore"):
        x[small] = log(expm1(x[small]))
    return x


def logexp1(x):
    x = np.asarray(x, dtype=float).copy()
    if np.any(np.isnan(x)):
        raise ValueError("NAs are not allowed in subscripted assignments")
    small = x < _CUTOFF
    x[small] = log(exp(x[small]) + 1)
    return x


def log_dnorm(x):
    return -(np.asarray(x, dtype=float) ** 2) / 2 - _LOG_2PI / 2


def dpnorm(x0, x1, log_p=True):
    x0, x1 = np.broadcast_arrays(np.asarray(x0, float), np.asarray(x1, float))
    x0, x1 = x0.copy(), x1.copy()
    if np.any(x0 >= x1):
        warnings.warn("some x0>=x1", RuntimeWarning, stacklevel=2)
    positive = (x0 > 0) & (x1 > 0)
    old = x0[positive].copy()
    x0[positive] = -x1[positive]
    x1[positive] = -old
    p0, p1 = log_pnorm(x0), log_pnorm(x1)
    result = p0 + logexm1(p1 - p0)
    return result if log_p else exp(result)


def ddnorm(x0, x1, a0=0, a1=0, s0=1, s1=1):
    """Return source log magnitude and sign of a density difference.

    The zero-sign assignments intentionally retain R's full-vector RHS
    replacement semantics; these differ from an elementwise simplification.
    """
    x0, x1, a0, a1, s0, s1 = np.broadcast_arrays(
        *[np.asarray(x, float) for x in (x0, x1, a0, a1, s0, s1)]
    )
    if np.any(x0 >= x1):
        warnings.warn("some x0>=x1", RuntimeWarning, stacklevel=2)
    p0, p1 = log_dnorm(x0) + a0, log_dnorm(x1) + a1
    result = p0.copy()
    sign = np.ones_like(p0)
    negative = (
        ((s1 < 0) & (s0 > 0))
        | ((s1 > 0) & (s0 > 0) & (p1 < p0))
        | ((s1 < 0) & (s0 < 0) & (p1 > p0))
    )
    sign[negative] = -1
    if np.any(np.isnan(p0) | np.isnan(p1)):
        raise ValueError("NAs are not allowed in subscripted assignments")
    swap = p0 > p1
    old = p1[swap].copy()
    p1[swap], p0[swap] = p0[swap], old
    same, opposite = s0 * s1 > 0, s0 * s1 < 0
    result[same] = p0[same] + logexm1(p1[same] - p0[same])
    result[opposite] = p0[opposite] + logexp1(p1[opposite] - p0[opposite])
    for mask, replacement, replacement_sign in ((s0 == 0, p1, s1), (s1 == 0, p0, -s0)):
        count = np.count_nonzero(mask)
        if count:
            if count % replacement.size:
                warnings.warn(
                    "number of items to replace is not a multiple of replacement length",
                    RuntimeWarning,
                    stacklevel=2,
                )
            result[mask] = replacement.ravel()[:count]
            sign[mask] = replacement_sign[mask]
    return result, sign


def log_pnorm(x):
    """R Mathlib pnorm_both lower log tail (Cody rational approximations).

    Preserves src/nmath/pnorm.c evaluation order, including the split square;
    tail rounding is amplified by cnorm's mixed fourth-order derivatives.
    Coefficients and branches: R Mathlib, Copyright R Core Team (GPL-2+).
    """
    x = np.asarray(x, dtype=float)
    y = np.abs(x)
    result = np.full_like(x, np.nan)
    a = (
        2.2352520354606839287,
        161.02823106855587881,
        1067.6894854603709582,
        18154.981253343561249,
        0.065682337918207449113,
    )
    b = (
        47.20258190468824187,
        976.09855173777669322,
        10260.932208618978205,
        45507.789335026729956,
    )
    c = (
        0.39894151208813466764,
        8.8831497943883759412,
        93.506656132177855979,
        597.27027639480026226,
        2494.5375852903726711,
        6848.1904505362823326,
        11602.651437647350124,
        9842.7148383839780218,
        1.0765576773720192317e-8,
    )
    d = (
        22.266688044328115691,
        235.38790178262499861,
        1519.377599407554805,
        6485.558298266760755,
        18615.571640885098091,
        34900.952721145977266,
        38912.003286093271411,
        19685.429676859990727,
    )
    p = (
        0.21589853405795699,
        0.1274011611602473639,
        0.022235277870649807,
        0.001421619193227893466,
        2.9112874951168792e-5,
        0.02307344176494017303,
    )
    q = (
        1.28426009614491121,
        0.468238212480865118,
        0.0659881378689285515,
        0.00378239633202758244,
        7.29751555083966205e-5,
    )
    small = y <= 0.67448975
    xx = x[small]
    square = xx * xx
    num, den = a[4] * square, square.copy()
    for i in range(3):
        num, den = (num + a[i]) * square, (den + b[i]) * square
    tiny = np.abs(xx) <= np.finfo(float).eps * 0.5
    num[tiny], den[tiny] = 0, 0
    temp = xx * (num + a[3]) / (den + b[3])
    result[small] = log(0.5 + temp)
    for central in (True, False):
        mask = (
            (y > 0.67448975) & (y <= np.sqrt(32))
            if central
            else (y > np.sqrt(32)) & (y < 1e170)
        )
        xx, yy = x[mask], y[mask]
        if central:
            num, den = c[8] * yy, yy.copy()
            for i in range(7):
                num, den = (num + c[i]) * yy, (den + d[i]) * yy
            temp = (num + c[7]) / (den + d[7])
            split_arg = yy
        else:
            with np.errstate(over="ignore"):
                square = 1 / (xx * xx)
            num, den = p[5] * square, square.copy()
            for i in range(4):
                num, den = (num + p[i]) * square, (den + q[i]) * square
            temp = square * (num + p[4]) / (den + q[4])
            temp = (0.398942280401432677939946059934 - temp) / yy
            split_arg = xx
        split = np.ldexp(np.trunc(np.ldexp(split_arg, 4)), -4)
        delta = (split_arg - split) * (split_arg + split)
        with np.errstate(over="ignore", under="ignore"):
            log_tail = -split * (split / 2) - delta / 2 + log(temp)
            positive = xx > 0
            log_tail[positive] = log1p(
                -exp(-split[positive] * (split[positive] / 2))
                * exp(-delta[positive] / 2)
                * temp[positive]
            )
        result[mask] = log_tail
    result[x >= 1e170] = -0.0
    result[x <= -1e170] = -np.inf
    return result
