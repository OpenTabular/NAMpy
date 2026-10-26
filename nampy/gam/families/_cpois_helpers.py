"""Censored-Poisson probabilities: mgcv efam.r::dppois and R ppois/pgamma.

R tail routines: Copyright (C) 1998 Ross Ihaka, 2000-2019 R Core Team,
2005-2006 Morten Welinder, 2005-2010 R Foundation. Python translation of
integer Poisson branches, GNU GPL version 2 or later, WITHOUT ANY WARRANTY.
See R_MATH_LICENSE.txt. mgcv dppois is by Simon N. Wood, distributed under GPL2.
"""

import math

import numpy as np
from scipy.special import gammainc, gammaincc

from ._cnorm_helpers import logexm1
from ._r_poisson_math import log_dpois_scalar
from ._scalar_math import exp, log


def log_dpois(y, mu):
    y, mu = np.broadcast_arrays(np.asarray(y, float), np.asarray(mu, float))
    return np.fromiter((log_dpois_scalar(float(q), float(m)) for q, m in
                        zip(y.flat, mu.flat, strict=True)), float, count=y.size).reshape(y.shape)


def log_ppois(y, mu, lower=True):
    """Log Poisson tails, retaining R's integer pgamma series in the tails.

    R ppois floors x+1e-7 and calls pgamma(lambda,x+1,!lower.tail).
    The integer-shape pd_lower_series and pd_upper_series branches are needed
    in log space: logging an already underflowed CDF loses finite tail values.
    The central branch uses SciPy's regularized incomplete gamma functions.
    """
    y, mu = np.broadcast_arrays(np.asarray(y, float), np.asarray(mu, float))
    result = np.empty_like(y)
    epsilon = np.finfo(float).eps
    for index in np.ndindex(y.shape):
        q, lam = float(y[index]), float(mu[index])
        if math.isnan(q) or math.isnan(lam) or lam < 0:
            value = np.nan
        elif q < 0:
            value = -np.inf if lower else 0.0
        elif lam == 0 or math.isinf(q):
            value = 0.0 if lower else -np.inf
        elif math.isinf(lam):
            value = -np.inf if lower else 0.0
        else:
            q = math.floor(q + 1e-7)
            alpha = q + 1.0
            if lam < 1:
                # R pgamma_smallx, lower gamma tail = upper Poisson tail.
                total, c, n = 0.0, alpha, 0
                while True:
                    n += 1
                    c *= -lam / n
                    term = c / (alpha + n)
                    total += term
                    if abs(term) <= epsilon * abs(total):
                        break
                upper_log = math.log1p(total) + alpha * math.log(lam) - math.lgamma(alpha + 1)
                value = math.log(-math.expm1(upper_log)) if lower else upper_log
            elif lam <= alpha - 1 and lam < 0.8 * (alpha + 50):
                # R pd_upper_series: sum starts at lambda/alpha.
                term = total = lam / alpha
                denominator = alpha
                while True:
                    denominator += 1
                    term *= lam / denominator
                    total += term
                    if term <= total * epsilon:
                        break
                upper_log = float(log_dpois(q, lam)) + math.log(total)
                value = math.log(-math.expm1(upper_log)) if lower else upper_log
            elif alpha - 1 < lam and alpha < 0.8 * (lam + 50):
                # Integer alpha: pd_lower_series has no fractional CF term.
                term, total, remaining = 1.0, 0.0, alpha - 1
                while remaining >= 1 and term > total * epsilon:
                    term *= remaining / lam
                    total += term
                    remaining -= 1
                lower_log = float(log_dpois(q, lam)) + math.log1p(total)
                value = lower_log if lower else math.log(-math.expm1(lower_log))
            else:
                probability = gammaincc(alpha, lam) if lower else gammainc(alpha, lam)
                value = math.log(probability)
        result[index] = value
    return result


def dppois(y0, y1, mu, log_p=True):
    """Direct branch port of efam.r::dppois, including negative endpoints."""
    y0, y1, mu = np.broadcast_arrays(np.asarray(y0, float), np.asarray(y1, float), np.asarray(mu, float))
    first, second = y0 < mu, y1 < mu
    p0, p1 = np.empty_like(mu), np.empty_like(mu)
    for lower in (True, False):
        mask = first == lower
        p0[mask] = log_ppois(y0[mask], mu[mask], lower)
        mask = second == lower
        p1[mask] = log_ppois(y1[mask], mu[mask], lower)
    value = np.zeros_like(mu)
    free = y1 >= 0
    value[~free] = -np.inf
    with np.errstate(all="ignore"):
        mask = second & (y0 < 0) & free
        value[mask] = p1[mask]
        free[mask] = False
        mask = ~second & (y0 < 0) & free
        value[mask] = log(1 - exp(p1[mask]))
        free[mask] = False
        mask = second & first & free
        value[mask] = p0[mask] + logexm1(p1[mask] - p0[mask])
        free[mask] = False
        mask = ~second & ~first & free
        value[mask] = p1[mask] + logexm1(p0[mask] - p1[mask])
        free[mask] = False
        mask = ~second & first & free
        value[mask] = log(1 - exp(p1[mask]) - exp(p0[mask]))
    return value if log_p else exp(value)
