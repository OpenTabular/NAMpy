"""Scalar integer dpois_raw port, R4.6 nmath (GPL-2-or-later).

Upstream dpois.c, bd0.c, stirlerr.c and pgamma.c by Catherine Loader,
Morten Welinder and the R Core Team. Copyright (C) 2000-2025 R Core Team,
2005-2006 Morten Welinder, 2005-2010 R Foundation, 2006-2019 R Core Team.
Python translation modified for integer Poisson arguments in NAMpy.
Redistribution/modification permitted under GNU GPL version 2 or later.
Provided WITHOUT ANY WARRANTY, including MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See R_MATH_LICENSE.txt for the complete license.
"""

import math

import numpy as np

from ._r_poisson_constants import BD0_SCALE, STIRLERR_INTEGERS


def _log1pmx(x):
    if x > 1 or x < -0.79149064:
        return math.log1p(x) - x
    r = x / (2 + x)
    y = r * r
    if abs(x) < 1e-2:
        return r * ((((2 / 9 * y + 2 / 7) * y + 2 / 5) * y + 2 / 3) * y - x)
    # pgamma.c::logcf(y,3,2,1e-14).
    c1, c2, c4 = 4.0, 5.0, 7.0
    a1, b1, b2 = c2, 3 * (c2 - 3 * y), 4 * y
    a2, b2 = c4 * c2 - b2, c4 * b1 - 3 * b2
    factor = 2.0**256
    while abs(a2 * b1 - a1 * b2) > abs(1e-14 * b1 * b2):
        c3 = c2 * c2 * y
        c2 += 2
        c4 += 2
        a1, b1 = c4 * a2 - c3 * a1, c4 * b2 - c3 * b1
        c3 = c1 * c1 * y
        c1 += 2
        c4 += 2
        a2, b2 = c4 * a1 - c3 * a2, c4 * b1 - c3 * b2
        if abs(b2) > factor:
            a1, b1, a2, b2 = (v / factor for v in (a1, b1, a2, b2))
        elif abs(b2) < 1 / factor:
            a1, b1, a2, b2 = (v * factor for v in (a1, b1, a2, b2))
    return r * (2 * y * a2 / b2 - x)


def _ebd0(x, mean):
    """bd0.c::ebd0, preserving its high/low accumulator and log table."""
    if x == mean:
        return 0.0, 0.0
    if x == 0:
        return mean, 0.0
    if mean == 0:
        return np.inf, 0.0
    ratio = mean / x
    if math.isinf(ratio):
        return mean, 0.0
    r, e = math.frexp(ratio)
    if math.log(2) * -e > 1 + np.finfo(float).max / x:
        return np.inf, 0.0
    i = math.floor((r - 0.5) * 256 + 0.5)
    f = math.floor(1024 / (0.5 + i / 256) + 0.5)
    try:
        fg = math.ldexp(f, -(e + 10))
    except OverflowError:
        return np.inf, 0.0
    high = low = 0.0

    def add(value):
        nonlocal high, low
        first = math.floor(value + 0.5)
        high += first
        low += value - first

    add(-x * _log1pmx((mean * fg - x) / x))
    if fg == 1:
        return high, low
    for j in range(4):
        add(x * BD0_SCALE[i][j])
        add(-x * BD0_SCALE[0][j] * e)
    add(mean)
    add(-mean * fg)
    return high, low


def _stirlerr_integer(n):
    if n <= 15:
        return STIRLERR_INTEGERS[int(n)]
    coefficients = (1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188,
                    691 / 360360, 1 / 156)
    if n > 15.7e6:
        order = 0
    elif n > 6180:
        order = 1
    elif n > 205:
        order = 2
    elif n > 86:
        order = 3
    elif n > 27:
        order = 4
    elif n > 23.5:
        order = 5
    else:
        order = 6
    value = coefficients[order]
    for coefficient in reversed(coefficients[:order]):
        value = coefficient - value / (n * n)
    return value / n


def log_dpois_scalar(x, mean):
    if math.isnan(x) or math.isnan(mean) or mean < 0:
        return np.nan
    if x < 0 or not math.isfinite(x):
        return -np.inf
    integer = round(x)
    if abs(x - integer) > 1e-7 * max(1, abs(x)):
        return -np.inf
    x = float(integer)
    if mean == 0:
        return 0.0 if x == 0 else -np.inf
    if not math.isfinite(mean):
        return -np.inf
    if x <= mean * np.finfo(float).tiny:
        return -mean
    if mean < x * np.finfo(float).tiny:
        return -mean + x * math.log(mean) - math.lgamma(x + 1)
    high, low = _ebd0(x, mean)
    low += _stirlerr_integer(x)
    if x >= 2.861117485757028e307:
        r = 2.5066282746310005 * math.sqrt(x)
        return -low - high - math.log(r)
    return -low - high - 0.5 * math.log(2 * math.pi * x)
