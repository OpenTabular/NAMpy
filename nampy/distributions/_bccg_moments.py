"""Numerical moments of the normalized BCCG law, with explicit existence gates.

For U=(Y/mu)**nu, U is N(1, (sigma*nu)**2) conditioned on U>0.
Negative powers have an algebraic endpoint, treated with QUADPACK's weighted
rule. Positive powers are integrated in log-response around the tilted mode.
These are probability integrals, independent of either fitting backend.
"""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import exprel, log_ndtr, logsumexp

from .laws import _lognormal_variance

_LOG_NORMAL = 0.5 * np.log(2 * np.pi)
_TOL = 2e-10


def _integrate(function, lower, upper, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(
                function, lower, upper, epsabs=0, epsrel=2e-11, limit=250, **kwargs
            )
        except (IntegrationWarning, OverflowError, ValueError) as exc:
            raise NotImplementedError(
                "BCCG moment quadrature did not converge in this numerical range."
            ) from exc
    if not np.isfinite(value) or value < 0 or error > _TOL * max(value, 1e-200):
        raise NotImplementedError(
            "BCCG moment quadrature could not certify relative accuracy in this numerical range."
        )
    return value


def _positive_log_moment(sigma, nu, k):
    # Stationary point in x=log(Y/mu): U**2-U=nu*sigma**2*(k+nu).
    c = nu * sigma**2 * (k + nu)
    if not np.isfinite(c):
        raise NotImplementedError(
            "BCCG tilted moment mode exceeds the supported numerical range."
        )
    u_minus_one = 2 * c / (1 + np.sqrt(1 + 4 * c))
    mode = np.log1p(u_minus_one) / nu
    u = 1 + u_minus_one
    width = sigma / np.sqrt(u * (2 * u - 1))
    z_mode = u_minus_one / (nu * sigma)
    peak = (k + nu) * mode - 0.5 * z_mode**2
    if not np.isfinite(peak) or width <= 0:
        raise NotImplementedError(
            "BCCG tilted moment scale exceeds the supported numerical range."
        )

    def integrand(v):
        # Stationarity cancels the linear term analytically, retaining narrow peaks.
        h = nu * width * v
        with np.errstate(over="ignore", invalid="ignore"):
            dz = u * width * v / sigma * exprel(h)
            # expm1(h)-h, divided by h, without cancellation at zero.
            remainder = (
                (
                    h / 2
                    + h * h / 6
                    + h**3 / 24
                    + h**4 / 120
                    + h**5 / 720
                    + h**6 / 5040
                    + h**7 / 40320
                )
                if abs(h) < 1e-3
                else exprel(h) - 1
            )
            exponent = -0.5 * dz**2 - z_mode * u * width * v / sigma * remainder
        return (
            0.0
            if np.isneginf(exponent) or np.isnan(exponent) and h > 0
            else float(np.exp(exponent))
        )

    integral = _integrate(integrand, -np.inf, 0) + _integrate(integrand, 0, np.inf)
    if not integral > 0:
        raise NotImplementedError(
            "BCCG tilted moment integral is outside the supported numerical range."
        )
    return (
        peak
        + np.log(width / sigma)
        + np.log(integral)
        - _LOG_NORMAL
        - log_ndtr(1 / (sigma * nu))
    )


def _negative_integral(sigma, nu, k, *, log_center=None):
    """Log raw moment, or log E[(Y/(mu*center)-1)**2] when centered."""
    t, r = -sigma * nu, k / nu
    epsilon = (nu + k) / nu
    if not np.isfinite(t) or t <= 0 or not epsilon > 0 or not r > -1:
        raise NotImplementedError(
            "BCCG finite-moment transform exceeds the supported numerical range."
        )
    a = 1 / t
    log_c = log_ndtr(a)
    cutoff = 0.5
    center = 0 if log_center is None else log_center
    endpoint_scale = (
        epsilon * np.log(cutoff)
        - np.log(t)
        - _LOG_NORMAL
        - log_c
        - 0.5 * ((1 - cutoff) / t) ** 2
        - (2 * center if log_center is not None else 0)
    )

    def endpoint(v):
        u = cutoff * v
        difference = (
            1.0 if log_center is None or u == 0 else -np.expm1(center - np.log(u) / nu)
        )
        exponent = ((1 - cutoff) ** 2 - (1 - u) ** 2) / (2 * t * t)
        return np.exp(exponent) * difference**2

    def normal(z):
        u = 1 + t * z
        if u < cutoff:
            return 0.0
        log_y = np.log1p(t * z) / nu
        factor = (
            np.exp(k * log_y) if log_center is None else np.expm1(log_y - center) ** 2
        )
        return factor * np.exp(-0.5 * z * z - _LOG_NORMAL - log_c)

    # Integrating reflected negative z from zero retains the normal peak even
    # when the finite lower bound is many orders of magnitude away.
    lower = (
        _integrate(lambda w: normal(-w), 0, (1 - cutoff) / t)
        if (1 - cutoff) / t < 8
        else _integrate(lambda w: normal(-w), 0, np.inf)
    )
    upper = _integrate(normal, 0, np.inf)
    logs = []
    for value in (lower, upper):
        logs.append(np.log(value) if value else -np.inf)
    # Omit the endpoint only when its analytic upper bound is below 1e-12
    # of the computed central integral. This also avoids resolving a narrow,
    # exponentially negligible boundary layer for very small sigma.
    endpoint_factor_bound = (
        0 if log_center is None else 2 * np.logaddexp(0, center - np.log(cutoff) / nu)
    )
    bound = endpoint_scale - np.log(epsilon) + endpoint_factor_bound
    if not bound <= logsumexp(logs) + np.log(1e-12):
        value = _integrate(endpoint, 0, 1, weight="alg", wvar=(r, 0))
        # QUADPACK receives rounded r. Restore the singular endpoint's exact
        # 1/epsilon integral, avoiding cancellation of 1+k/nu near nu=-k.
        value += endpoint(0.0) * ((1 + r) - epsilon) / ((1 + r) * epsilon)
        logs.append(endpoint_scale + np.log(value))
    result = logsumexp(logs)
    if not np.isfinite(result):
        raise NotImplementedError(
            "BCCG moment is outside the supported integration range."
        )
    return result


def _small_variance(sigma, nu, log_mean):
    if nu < 0:
        if sigma < np.sqrt(np.finfo(float).eps):
            raise NotImplementedError(
                'BCCG negative-power centered variance at sigma below sqrt(machine epsilon) '
                'has unvalidated mean precision in this numerical range.'
            )
        return 2 * log_mean + _negative_integral(sigma, nu, 2, log_center=log_mean)
    t = sigma * nu
    a, log_c = 1 / t, log_ndtr(1 / t)

    def normal(z, degree):
        if z <= -a:
            return 0.0
        v = t * z
        ratio = np.log1p(v) / v if v else 1.0
        log_y = sigma * z * ratio
        # D=(Y/mu-1)/sigma, retaining a representable scale even when sigma**2
        # underflows. Center D directly instead of subtracting raw Y moments.
        deviation = z * ratio * exprel(log_y)
        return abs(deviation) ** degree * np.exp(-0.5 * z * z - _LOG_NORMAL - log_c)

    halves = []
    for degree in (1, 2):
        lower = _integrate(
            lambda z, degree=degree: normal(-z, degree), 0, a if a < 8 else np.inf
        )
        upper = _integrate(lambda z, degree=degree: normal(z, degree), 0, np.inf)
        halves.append((lower, upper))
    mean = halves[0][1] - halves[0][0]
    second = sum(halves[1])
    value = second - mean * mean
    if not value > 0 or value < 1e-6 * second:
        raise NotImplementedError(
            "BCCG centered variance is outside the supported integration range."
        )
    return 2 * np.log(sigma) + np.log(value)


def moments(mu, sigma, nu, operation):
    result = np.empty_like(mu)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for row, (m, s, n) in enumerate(zip(mu, sigma, nu, strict=True)):
            if n == 0:
                result[row] = (
                    np.exp(np.log(m) + s * s / 2)
                    if operation == "mean"
                    else _lognormal_variance(np.log(m), s)
                )
                continue
            if -1 <= n < 0:
                result[row] = np.inf if operation == "mean" else np.nan
                continue
            if operation == "variance" and -2 <= n < -1:
                result[row] = np.inf
                continue
            log_m1 = (
                _positive_log_moment(s, n, 1) if n > 0 else _negative_integral(s, n, 1)
            )
            if operation == "mean":
                result[row] = np.exp(np.log(m) + log_m1)
                continue
            log_m2 = (
                _positive_log_moment(s, n, 2) if n > 0 else _negative_integral(s, n, 2)
            )
            gap = log_m2 - 2 * log_m1
            log_variance = (
                log_m2 + np.log(-np.expm1(-gap))
                if gap > 1e-5
                else _small_variance(s, n, log_m1)
            )
            result[row] = np.exp(2 * np.log(m) + log_variance)
    return result
