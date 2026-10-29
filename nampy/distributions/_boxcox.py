"""Shared positive-support Box-Cox probabilities for symmetric latent laws.

This owner supplies the response Jacobian, truncation, stable interval masses,
quantile inversion and sampling. BCCG retains its existing numerical path.
No estimator, training backend or fitting derivative is imported here.
"""

from typing import ClassVar

import numpy as np

from ._boxcox_latent import (
    _SymmetricLatent,
    checked_integral,
    checked_root,
    log_complement,
)
from .bccg import _log_ratio
from .contracts import DistributionSpec

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
    }
)


def _transform(log_ratio, sigma, nu):
    """Signed log magnitude of z=expm1(nu*log_ratio)/(nu*sigma)."""
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = nu * log_ratio
        result = np.log(np.abs(log_ratio)) - np.log(sigma)
        active = x != 0
        xx = x[active]
        positive = xx > 0
        log_abs = np.empty_like(xx)
        log_abs[positive] = xx[positive] + np.log(-np.expm1(-xx[positive]))
        log_abs[~positive] = np.log(-np.expm1(xx[~positive]))
        result[active] = log_abs - np.log(np.abs(nu[active])) - np.log(sigma[active])
    if np.any(~np.isfinite(x)):
        raise NotImplementedError(
            "Box-Cox transform exceeds the supported numerical range."
        )
    return np.sign(log_ratio), result


def _upper(log_a, log_delta):
    """Signed log magnitude of -a+delta, retaining the explicit interval width."""
    if log_delta == log_a:
        return 0.0, -np.inf
    if log_delta < log_a:
        return -1.0, log_a + np.log(-np.expm1(log_delta - log_a))
    return 1.0, log_delta + np.log(-np.expm1(log_a - log_delta))


def _interval_mass(latent, tau, log_a, log_delta, sign=None, log_abs=None):
    """log integral_{-a}^{-a+delta} f(z) dz, with explicit log width.

    Use log-CDF differences where separated. Otherwise integrate on [0,1]
    with a bounded density factor and an explicit zero crossing for PE cusps.
    No normal/t or other replacement distribution is used in the boundary.
    """
    if sign is None:
        sign, log_abs = _upper(log_a, log_delta)
    lo = float(latent.logcdf(-1.0, log_a, tau))
    hi = float(latent.logcdf(sign, log_abs, tau))
    if np.isneginf(lo) and np.isfinite(hi):
        return hi
    if hi - lo > 0.5:
        return hi + float(log_complement(lo - hi))
    # A peaked unresolved interval is explicitly unavailable rather than
    # silently accepting a quadrature result that has missed its mass.
    if log_delta <= log_a:
        ratio = np.exp(log_delta - log_a)
        peak = float(latent.logpdf(log_abs, tau))
        start = float(latent.logpdf(log_a, tau))
        if np.isneginf(peak):
            return -np.inf
        variation = peak - start

        def integrand(s):
            position = log_a + np.log1p(-ratio * s)
            return float(np.exp(latent.logpdf(position, tau) - peak))

        points = []
    else:
        ratio = np.exp(log_a - log_delta)
        peak = float(latent.logpdf(-np.inf, tau))
        start = min(
            float(latent.logpdf(log_a, tau)), float(latent.logpdf(log_abs, tau))
        )
        variation = peak - start

        def integrand(s):
            with np.errstate(divide="ignore"):
                position = log_delta + np.log(abs(s - ratio))
            return float(np.exp(latent.logpdf(position, tau) - peak))

        points = [ratio] if 0 < ratio < 1 else []
    if not np.isfinite(variation) or variation > 40:
        raise NotImplementedError(
            "Box-Cox boundary interval exceeds the supported integration range."
        )
    integral = checked_integral(integrand, points=points)
    return log_delta + peak + np.log(integral)


def _quantile_rows(latent, mu, sigma, nu, tau, probabilities):
    mu, sigma, nu, tau, p = np.broadcast_arrays(
        mu[:, None], sigma[:, None], nu[:, None], tau[:, None], probabilities
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        log_t = np.log(sigma) + np.log(np.abs(nu))
        log_a = -log_t
        log_c = latent.logcdf(1.0, log_a, tau)
        log_tail = np.where(nu > 0, np.log1p(-p), np.log(p)) + log_c
        sign_z, log_z = latent.isf_log(log_tail, tau)
        sign_z = np.where(nu > 0, sign_z, -sign_z)
        b = log_t + log_z
        same = np.sign(nu) * sign_z >= 0
        # Work in log magnitude before dividing by nu. This retains the
        # removable limit when nu*sigma*z itself would underflow.
        negative_log_u = np.where(
            b < -np.log(2.0), np.log1p(-np.exp(b)), np.log(-np.expm1(b))
        )
        log_change = np.where(same, np.log(np.logaddexp(0, b)), np.log(-negative_log_u))
        log_change = np.where(b < -36, b, log_change)
        offset = np.where(same, np.sign(nu), -np.sign(nu)) * np.exp(
            log_change - np.log(np.abs(nu))
        )
        zero = nu == 0
        offset[zero] = sign_z[zero] * np.exp(np.log(sigma[zero]) + log_z[zero])
        log_y = np.log(mu) + offset
        boundary = (~zero) & (~same) & (b > np.log(0.75)) & (p > 0) & (p < 1)
        for index in zip(*np.nonzero(boundary), strict=True):
            aa, tt = float(log_a[index]), float(tau[index])
            target = (
                np.log(p[index]) if nu[index] > 0 else np.log1p(-p[index])
            ) + log_c[index]
            maximum_density = float(latent.logpdf(-np.inf, tt))
            lower, upper = target - maximum_density, aa + np.log(2.0)

            def equation(ld, aa=aa, tt=tt, target=target):
                return _interval_mass(latent, tt, aa, ld) - target

            # These exact density/interval bounds contain a boundary quantile.
            if not np.isfinite(lower) or equation(lower) > 1e-10 or equation(upper) < 0:
                raise NotImplementedError(
                    "Box-Cox boundary quantile exceeds the supported numerical range."
                )
            width = checked_root(equation, lower, upper)
            log_y[index] = np.log(mu[index]) + (log_t[index] + width) / nu[index]
        output = np.exp(log_y)
        output[p == 0], output[p == 1] = 0, np.inf
    if np.any(np.isnan(output)):
        raise NotImplementedError(
            "Box-Cox quantile exceeds the supported numerical range."
        )
    return output


class SymmetricBoxCoxLaw:
    """Private operation owner; concrete laws declare spec and latent law."""

    spec: ClassVar[DistributionSpec]
    latent: ClassVar[_SymmetricLatent]

    def validate_parameters(self, parameters, *, context):
        if any(np.any(parameters[name] <= 0) for name in ("mu", "sigma", "tau")):
            raise ValueError(
                f"{self.spec.name} requires positive mu, sigma and tau; nu is an unrestricted finite power."
            )
        self.latent.validate(parameters["tau"])

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        if operation not in self.spec.operations:
            raise NotImplementedError(
                f"{self.spec.name} does not support {operation!r}."
            )
        mu, sigma, nu, tau = (parameters[name] for name in self.spec.parameter_names)
        if operation in {"mean", "variance"}:
            from ._boxcox_moments import moments
            return moments(self.latent,self.spec.name,parameters,operation)
        if operation == "quantile":
            return _quantile_rows(
                self.latent,
                mu,
                sigma,
                nu,
                tau,
                np.broadcast_to(values, (len(mu), len(values))),
            )
        if operation == "sample":
            p = rng.random((len(mu), n_draws))
            while np.any(p == 0):
                p[p == 0] = rng.random(np.count_nonzero(p == 0))
            return _quantile_rows(self.latent, mu, sigma, nu, tau, p).T
        output = np.full(len(mu), np.nan)
        inside = (values > 0) & np.isfinite(values)
        if np.any(inside):
            y, mm, ss, nn, tt = (
                values[inside],
                mu[inside],
                sigma[inside],
                nu[inside],
                tau[inside],
            )
            lr = _log_ratio(y, mm)
            sign_z, log_z = _transform(lr, ss, nn)
            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                log_t = np.log(ss) + np.log(np.abs(nn))
                log_a = -log_t
                log_c = self.latent.logcdf(1.0, log_a, tt)
                if operation == "log_prob":
                    result = (
                        nn * lr
                        - np.log(ss)
                        - np.log(y)
                        + self.latent.logpdf(log_z, tt)
                        - log_c
                    )
                else:
                    zero = nn == 0
                    log_cdf = np.empty(len(y))
                    log_sf = np.empty(len(y))
                    log_cdf[zero] = self.latent.logcdf(
                        sign_z[zero], log_z[zero], tt[zero]
                    )
                    log_sf[zero] = self.latent.logcdf(
                        -sign_z[zero], log_z[zero], tt[zero]
                    )
                    for j in np.flatnonzero(~zero):
                        sign = np.sign(nn[j]) * sign_z[j]
                        interval = (
                            _interval_mass(
                                self.latent,
                                tt[j],
                                log_a[j],
                                nn[j] * lr[j] - log_t[j],
                                sign,
                                log_z[j],
                            )
                            - log_c[j]
                        )
                        complement = (
                            float(self.latent.logcdf(-sign, log_z[j], tt[j])) - log_c[j]
                        )
                        log_cdf[j], log_sf[j] = (
                            (interval, complement)
                            if nn[j] > 0
                            else (complement, interval)
                        )
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
                f"{self.spec.name} probability exceeds the supported numerical range."
            )
        return output
