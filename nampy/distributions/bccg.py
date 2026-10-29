"""Normalized Box-Cox Cole-Green probabilities, independent of fitting engines.

The density and parameterization follow gamlss.dist/R/BCCG.R d/p/qBCCG.
Log-scale evaluation, normal interval masses and quantile inversion preserve
that law where the source's direct powers/CDF subtraction lose precision.

References
----------
- Cole, T. J., and Green, P. J. (1992). Smoothing reference centile curves: the LMS
  method and penalized likelihood.
  https://doi.org/10.1002/sim.4780111005
- Rigby, R. A., and Stasinopoulos, D. M. (2005). Generalized additive models for
  location, scale and shape.
  https://doi.org/10.1111/j.1467-9876.2005.00510.x
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcx, exprel, log_ndtr, logsumexp, ndtri_exp, roots_legendre

from ._bccg_moments import moments
from .contracts import DistributionSpec

_LOG_2PI = np.log(2 * np.pi)
_NODES, _WEIGHTS = roots_legendre(16)
_NODES, _WEIGHTS = (_NODES + 1) / 2, _WEIGHTS / 2
_OPERATIONS = frozenset(
    {
        "log_prob",
        "cdf",
        "cdf_left",
        "log_cdf",
        "survival",
        "log_survival",
        "quantile",
        "sample",
        "mean",
        "variance",
    }
)


def _log_ratio(y, mu):
    """Retain differences near mu without forming an overflowing y/mu ratio."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative = (y - mu) / mu
        close = np.abs(relative) < 0.5
        result = np.log(y) - np.log(mu)
        result[close] = np.log1p(relative[close])
    return result


def _transformed(log_ratio, sigma, nu):
    """Stable z=expm1(nu*log_ratio)/(nu*sigma), including the removable limit."""
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = nu * log_ratio
        denominator = nu * sigma
        numerator = np.expm1(x)
        direct = (
            (denominator != 0)
            & np.isfinite(denominator)
            & np.isfinite(numerator)
            & (x != 0)
        )
        result = np.empty_like(x)
        result[direct] = numerator[direct] / denominator[direct]
        limit = x == 0
        result[limit] = log_ratio[limit] / sigma[limit]
        other = ~direct & ~limit
        xx = x[other]
        log_abs = np.empty_like(xx)
        positive = xx > 0
        log_abs[positive] = xx[positive] + np.log(-np.expm1(-xx[positive]))
        log_abs[~positive] = np.log(-np.expm1(xx[~positive]))
        log_t = np.log(np.abs(nu[other])) + np.log(sigma[other])
        result[other] = np.sign(log_ratio[other]) * np.exp(log_abs - log_t)
        base = log_ratio / sigma
        small = (np.abs(x) <= 1) & np.isfinite(base)
        result[small] = base[small] * exprel(x[small])
    return result


def _log_lower_mass(a, log_delta, upper=None):
    """log[Phi(-a+delta)-Phi(-a)] from an explicit log interval width.

    Near the boundary use phi(a)*delta times the integral on [0,1] of
    exp(a*delta*s-delta**2*s**2/2). Sixteen-point Gaussian quadrature operates
    only where delta*(1+a)<=0.1, bounding the exponent variation by 0.1.
    Elsewhere use scaled-erfc Mills ratios or log-CDF differences. No
    distributional tail approximation or replacement law is substituted.
    """
    a, log_delta = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(log_delta, dtype=float)
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        delta = np.exp(log_delta)
        upper = delta - a if upper is None else np.broadcast_to(upper, a.shape)
        log_hi, log_lo = log_ndtr(upper), log_ndtr(-a)
        output = np.empty_like(log_hi)
        small = log_delta + np.log1p(a) <= np.log(0.1)
        if np.any(small):
            aa, ld, dd = a[small], log_delta[small], delta[small]
            ad = np.exp(np.log(aa) + ld)
            exponent = ad[:, None] * _NODES - 0.5 * dd[:, None] ** 2 * _NODES**2
            integral = logsumexp(exponent, b=_WEIGHTS, axis=1)
            output[small] = ld - 0.5 * aa**2 - 0.5 * _LOG_2PI + integral
        unbounded = ~small & np.isneginf(log_lo)
        output[unbounded] = log_hi[unbounded]
        vanished = ~small & ~unbounded & np.isneginf(log_hi)
        output[vanished] = -np.inf
        negative = ~small & ~unbounded & ~vanished & (upper < 0)
        if np.any(negative):
            aa, dd, uu = a[negative], delta[negative], upper[negative]
            # Difference of log-CDFs without subtracting their large squares.
            difference = (
                dd * (aa - 0.5 * dd)
                + np.log(erfcx(-uu / np.sqrt(2)))
                - np.log(erfcx(aa / np.sqrt(2)))
            )
            output[negative] = log_hi[negative] + np.log(-np.expm1(-difference))
        remaining = ~small & ~unbounded & ~vanished & ~negative
        output[remaining] = log_hi[remaining] + np.log(
            -np.expm1(log_lo[remaining] - log_hi[remaining])
        )
    if np.any(np.isnan(output)):
        raise NotImplementedError(
            "BCCG normal interval is outside the supported floating-point range; no tail clipping is substituted."
        )
    return np.minimum(output, 0)


def _quantile_rows(mu, sigma, nu, probabilities):
    """Rowwise probability grids, also used by inverse-transform sampling."""
    mu, sigma, nu, p = np.broadcast_arrays(
        mu[:, None], sigma[:, None], nu[:, None], probabilities
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        log_t = np.log(np.abs(nu)) + np.log(sigma)
        a = np.exp(-log_t)
        log_c = log_ndtr(a)
        positive = nu > 0
        z = np.where(
            positive, -ndtri_exp(np.log1p(-p) + log_c), ndtri_exp(np.log(p) + log_c)
        )
        signed_t = nu * sigma
        u = signed_t * z
        log_y = np.log(mu) + np.log1p(u) / nu
        zero = nu == 0
        log_y[zero] = np.log(mu[zero]) + sigma[zero] * z[zero]
        # Exact removable limit when the product rounds to zero.
        limit = ~zero & (u == 0)
        log_y[limit] = np.log(mu[limit]) + sigma[limit] * z[limit]
        base = sigma * z
        small = (np.abs(u) <= 0.5) & np.isfinite(base)
        ratio = np.ones(np.count_nonzero(small))
        nonzero = u[small] != 0
        ratio[nonzero] = np.log1p(u[small][nonzero]) / u[small][nonzero]
        log_y[small] = np.log(mu[small]) + base[small] * ratio
        # Avoid exp overflow in positive 1+u; evaluate its logarithm directly.
        large = ~zero & np.isposinf(u)
        log_u = log_t[large] + np.log(np.abs(z[large]))
        log_y[large] = np.log(mu[large]) + np.logaddexp(0, log_u) / nu[large]
        # Near the finite normal boundary, z cannot retain a small 1+u.
        # Invert the same normalized interval in log width, not a rounded z.
        boundary = ~zero & (p > 0) & (p < 1) & ((1 + u) < 0.25)
        for index in zip(*np.nonzero(boundary), strict=True):
            aa = a[index]
            if not np.isfinite(aa):
                raise NotImplementedError(
                    "BCCG boundary quantile is outside the supported floating-point range."
                )
            target = (
                np.log(p[index]) if positive[index] else np.log1p(-p[index])
            ) + log_c[index]
            lower = target + 0.5 * _LOG_2PI
            upper = np.log(aa + max(0.0, np.sign(nu[index]) * z[index]) + 1)

            def equation(ld, aa=aa, target=target):
                return float(
                    _log_lower_mass(np.array([aa]), np.array([ld]))[0] - target
                )

            ld = brentq(
                equation, lower, upper, xtol=5e-14, rtol=4 * np.finfo(float).eps
            )
            log_y[index] = np.log(mu[index]) + (log_t[index] + ld) / nu[index]
        output = np.exp(log_y)
        output[p == 0] = 0
        output[p == 1] = np.inf
    if np.any(np.isnan(output)):
        raise NotImplementedError(
            "BCCG quantiles exceed the supported floating-point range; no substitute distribution is used."
        )
    return output


@dataclass(frozen=True)
class BCCGLaw:
    """Box-Cox Cole-Green law with mu>0, sigma>0 and real Box-Cox power nu.

    z=((y/mu)**nu-1)/(nu*sigma), or log(y/mu)/sigma at nu=0.
    z is standard normal conditioned on z>-1/(nu*sigma) for nu>0,
    or z<-1/(nu*sigma) for nu<0. Density includes the response Jacobian
    and divides by Phi(1/(sigma*abs(nu))). nu=0 is lognormal with
    log-location log(mu) and log-SD sigma. mu is not generally the mean
    or exact median. This is not mgcv's censored bcg likelihood.

    Mean is infinite for -1<=nu<0; variance is undefined (NaN) there and
    infinite for -2<=nu<-1. Else moments use checked numerical integration,
    with an exact lognormal branch at nu=0. Statistical bccg_lss fitting is
    fixed-smoothing or automatic joint REML; neural fitting uses the normalized likelihood. Finite negative-power
    variance below sigma=sqrt(machine eps)
    is unsupported because centering precision is not validated there.
    Outcome quantiles/draws retain float64
    underflow/overflow limits. Normal boundary intervals use controlled
    Gaussian quadrature; inverse boundary quantiles solve in log width.
    """

    spec = DistributionSpec(
        "bccg",
        ("mu", "sigma", "nu"),
        _OPERATIONS,
        "positive_truncated_box_cox_cole_green_v1",
        "continuous",
    )

    def validate_parameters(self, parameters, *, context):
        if np.any(parameters["mu"] <= 0) or np.any(parameters["sigma"] <= 0):
            raise ValueError(
                "BCCG mu and sigma must be positive; nu is an unrestricted finite power."
            )

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        if operation not in self.spec.operations:
            raise NotImplementedError(f"BCCG does not support {operation!r}.")
        mu, sigma, nu = (parameters[name] for name in self.spec.parameter_names)
        if operation in {"mean", "variance"}:
            return moments(mu, sigma, nu, operation)
        if operation == "quantile":
            return _quantile_rows(
                mu, sigma, nu, np.broadcast_to(values, (len(mu), len(values)))
            )
        if operation == "sample":
            probabilities = rng.random(size=(len(mu), n_draws))
            while np.any(probabilities == 0):
                empty = probabilities == 0
                probabilities[empty] = rng.random(size=int(empty.sum()))
            return _quantile_rows(mu, sigma, nu, probabilities).T
        output = np.full(len(mu), np.nan)
        inside = (values > 0) & np.isfinite(values)
        if np.any(inside):
            y, mm, ss, nn = values[inside], mu[inside], sigma[inside], nu[inside]
            lr = _log_ratio(y, mm)
            z = _transformed(lr, ss, nn)
            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                log_t = np.log(np.abs(nn)) + np.log(ss)
                a = np.exp(-log_t)
                log_c = log_ndtr(a)
                if operation == "log_prob":
                    result = (
                        nn * lr
                        - np.log(ss)
                        - 0.5 * z * z
                        - np.log(y)
                        - 0.5 * _LOG_2PI
                        - log_c
                    )
                    result[~np.isfinite(z)] = -np.inf
                else:
                    zero = nn == 0
                    log_cdf, log_sf = np.empty(len(y)), np.empty(len(y))
                    log_cdf[zero], log_sf[zero] = log_ndtr(z[zero]), log_ndtr(-z[zero])
                    nonzero = ~zero
                    v = np.sign(nn[nonzero]) * z[nonzero]
                    lower_mass = (
                        _log_lower_mass(
                            a[nonzero], nn[nonzero] * lr[nonzero] - log_t[nonzero], v
                        )
                        - log_c[nonzero]
                    )
                    complement = log_ndtr(-v) - log_c[nonzero]
                    is_positive = nn[nonzero] > 0
                    log_cdf[nonzero] = np.where(is_positive, lower_mass, complement)
                    log_sf[nonzero] = np.where(is_positive, complement, lower_mass)
                    log_cdf, log_sf = np.minimum(log_cdf, 0), np.minimum(log_sf, 0)
                    result = (
                        log_cdf
                        if operation == "log_cdf"
                        else log_sf
                        if operation == "log_survival"
                        else np.exp(log_sf)
                        if operation == "survival"
                        else np.exp(log_cdf)
                    )
            output[inside] = result
        if operation == "log_prob":
            output[(values <= 0) | np.isposinf(values)] = -np.inf
        elif operation in {"cdf", "cdf_left", "log_cdf"}:
            output[values <= 0] = -np.inf if operation == "log_cdf" else 0
            output[np.isposinf(values)] = 0 if operation == "log_cdf" else 1
        else:
            output[values <= 0] = 0 if operation == "log_survival" else 1
            output[np.isposinf(values)] = -np.inf if operation == "log_survival" else 0
        if np.any(np.isnan(output[~np.isnan(values)])):
            raise NotImplementedError(
                "BCCG evaluation exceeded the supported floating-point range."
            )
        return output
