"""Positive raw moments of normalized symmetric Box-Cox laws.

Moment existence is checked analytically before quadrature. Integration failure
is explicit; no tail truncation or replacement family is used.
"""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gammaln


def _integral(function, lower, upper, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(
                function, lower, upper, epsabs=0, epsrel=2e-10, limit=250, **kwargs
            )
        except (IntegrationWarning, ValueError, OverflowError) as exc:
            raise NotImplementedError(
                "Box-Cox moment quadrature could not certify this numerical range."
            ) from exc
    if not np.isfinite(value) or value < 0 or error > 2e-8 * max(value, 1e-200):
        raise NotImplementedError(
            "Box-Cox moment quadrature could not certify relative accuracy."
        )
    return value


def raw_moment(latent, kind, mu, sigma, nu, tau, order):
    """E[Y**order], returning infinity for a mathematically divergent moment."""
    if nu < 0 and nu >= -order:
        return np.inf
    if kind == "bct" and (nu == 0 or nu > 0 and order / nu >= tau):
        return np.inf
    log_c = 0.5 * (gammaln(1 / tau) - gammaln(3 / tau)) if kind == "bcpe" else 0.0
    if nu == 0 and (tau < 1 or tau == 1 and order * sigma >= np.exp(-log_c)):
        return np.inf

    def exponential(value):
        if value > np.log(np.finfo(float).max):
            raise NotImplementedError(
                "Finite Box-Cox moment exceeds the supported integration range."
            )
        return float(np.exp(value))

    if nu == 0:

        def integrand(z):
            return exponential(
                order * sigma * z + latent.logpdf(np.log(abs(z)) if z else -np.inf, tau)
            )

        moment = _integral(integrand, -np.inf, 0) + _integral(integrand, 0, np.inf)
    else:
        width = abs(sigma * nu)
        power = order / nu
        normalizer = float(latent.logcdf(1.0, -np.log(width), tau))

        def density(u):
            log_z = np.log(abs(u - 1)) - np.log(width) if u != 1 else -np.inf
            return float(latent.logpdf(log_z, tau)) - np.log(width) - normalizer

        moment = _integral(
            lambda u: exponential(density(u)), 0, 1, weight="alg", wvar=(power, 0)
        )
        moment += _integral(
            lambda u: exponential(power * np.log(u) + density(u)), 1, np.inf
        )
    with np.errstate(over="ignore"):
        return float(mu**order * moment)


def moments(latent, kind, parameters, operation):
    result = []
    for mu, sigma, nu, tau in zip(
        *(parameters[k] for k in ("mu", "sigma", "nu", "tau")), strict=True
    ):
        first = raw_moment(latent, kind, mu, sigma, nu, tau, 1)
        if operation == "mean":
            result.append(first)
        else:
            second = raw_moment(latent, kind, mu, sigma, nu, tau, 2)
            if np.isinf(first):
                result.append(np.nan)
            elif np.isinf(second):
                result.append(np.inf)
            else:
                variance = second - first * first
                if variance < 0:
                    raise NotImplementedError(
                        "Box-Cox variance cancellation exceeds the supported numerical range."
                    )
                result.append(variance)
    return np.asarray(result)
