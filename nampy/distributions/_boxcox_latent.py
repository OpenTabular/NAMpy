"""Symmetric latent laws for normalized positive Box-Cox distributions.

BCT uses an ordinary Student-t scale; BCPE uses a variance-one generalized
normal, with alpha=exp((lgamma(1/tau)-lgamma(3/tau))/2). These match the
latent laws in gamlss.dist/R/BCT.R and BCPE.R, without the BCT density-only
Gaussian switch. Checked integrals and continued fractions evaluate the same
law when direct special-function probabilities underflow.
"""

import warnings

import numpy as np
from scipy import special, stats
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

_LOG2 = np.log(2.0)
_LOG_HALF = -_LOG2


def _indices(mask):
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return [()] if mask else []
    return zip(*np.nonzero(mask), strict=True)


def log_complement(log_p):
    """log(1-exp(log_p)), retaining tiny complements near log_p=0."""
    log_p = np.asarray(log_p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            log_p < -_LOG2, np.log1p(-np.exp(log_p)), np.log(-np.expm1(log_p))
        )


def checked_integral(function, *, points=None):
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(
                function,
                0,
                1 if points is not None else np.inf,
                points=points,
                epsabs=0,
                epsrel=2e-12,
                limit=300,
            )
        except (IntegrationWarning, OverflowError, ValueError) as exc:
            raise NotImplementedError(
                "Box-Cox probability integral did not converge in this numerical range."
            ) from exc
    if not value > 0 or not np.isfinite(value) or error > 2e-10 * value:
        raise NotImplementedError(
            "Box-Cox probability integral could not certify accuracy in this numerical range."
        )
    return value


def checked_root(function, lower, upper):
    try:
        return brentq(
            function,
            lower,
            upper,
            xtol=5e-13,
            rtol=4 * np.finfo(float).eps,
            maxiter=200,
        )
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise NotImplementedError(
            "Box-Cox probability inversion did not converge in this numerical range."
        ) from exc


def _gamma_log_survival(shape, x):
    shape, x = np.broadcast_arrays(shape, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.array(np.log(special.gammaincc(shape, x)), copy=True)
    # Upper incomplete-gamma continued fraction (modified Lentz), evaluated
    # after its exponential prefactor has underflowed in ordinary probability.
    for index in _indices(np.isneginf(result) & np.isfinite(x)):
        a, z = float(shape[index]), float(x[index])
        if z < a + 1 or not np.isfinite(a):
            raise NotImplementedError(
                "BCPE incomplete-gamma tail is outside the supported numerical range."
            )
        tiny = float(np.finfo(float).tiny)
        b = z + 1 - a
        c, d = 1 / tiny, 1 / b
        h = d
        for degree in range(1, 10001):
            numerator = -degree * (degree - a)
            b += 2
            d = b + numerator * d
            c = b + numerator / c
            d = np.copysign(max(abs(d), tiny), d)
            c = np.copysign(max(abs(c), tiny), c)
            d = 1 / d
            factor = d * c
            h *= factor
            if abs(factor - 1) < 4 * np.finfo(float).eps:
                break
        else:
            raise NotImplementedError(
                "BCPE incomplete-gamma continued fraction did not converge."
            )
        result[index] = -z + a * np.log(z) - special.gammaln(a) + np.log(h)
    return result


class _SymmetricLatent:
    def validate(self, tau):
        raise NotImplementedError

    def logpdf(self, log_abs, tau):
        raise NotImplementedError

    def logsf_abs(self, log_abs, tau):
        raise NotImplementedError

    def _inverse_tail(self, log_p, tau):
        raise NotImplementedError

    def logcdf(self, sign, log_abs, tau):
        tail = self.logsf_abs(log_abs, tau)
        return np.where(sign < 0, tail, log_complement(tail))

    def isf_log(self, log_p, tau):
        log_p, tau = np.broadcast_arrays(log_p, tau)
        negative = log_p > _LOG_HALF
        tail = np.where(negative, log_complement(log_p), log_p)
        log_abs = self._inverse_tail(tail, tau)
        sign = np.where(negative, -1.0, 1.0)
        median = log_p == _LOG_HALF
        return np.where(median, 0.0, sign), np.where(median, -np.inf, log_abs)


class StudentLatent(_SymmetricLatent):
    def validate(self, tau):
        if np.any(tau / 2 == 0):
            raise NotImplementedError(
                "BCT degrees of freedom are below the supported numerical range."
            )

    def logpdf(self, log_abs, tau):
        b = log_abs - 0.5 * np.log(tau)
        # Split the softplus to avoid overflowing z**2 or 2*log(abs(z)).
        return (
            -0.5 * np.log(tau)
            - special.betaln(tau / 2, 0.5)
            - (tau + 1) * np.maximum(b, 0)
            - 0.5 * (tau + 1) * np.log1p(np.exp(-2 * np.abs(b)))
        )

    def logsf_abs(self, log_abs, tau):
        log_abs, tau = np.broadcast_arrays(log_abs, tau)
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            result = np.array(stats.t.logsf(np.exp(log_abs), tau), copy=True)
            for index in _indices(np.isneginf(result) & np.isfinite(log_abs)):
                lz, df = float(log_abs[index]), float(tau[index])
                b, a = lz - 0.5 * np.log(df), df / 2
                if b >= 0:
                    h = np.exp(-np.logaddexp(0, 2 * b))
                    hyper = special.hyp2f1(a, 0.5, a + 1, h)
                    result[index] = (
                        -_LOG2
                        - df * b
                        - a * np.log1p(np.exp(-2 * b))
                        - np.log(a)
                        - special.betaln(a, 0.5)
                        + np.log(hyper)
                    )
                else:
                    # t=z*exp(s) turns the tail into a nonnegative integral.
                    # Normalize its local decay rate so the peak is retained.
                    w = np.exp(-np.logaddexp(0, np.log(df) - 2 * lz))
                    rate = (df + 1) * w - 1
                    if not rate > 0:
                        raise NotImplementedError(
                            "BCT tail is outside the supported numerical range."
                        )

                    def integrand(v, w=w, df=df, rate=rate):
                        s = v / rate
                        log_increment = np.log(w) + 2 * s + np.log(-np.expm1(-2 * s))
                        if log_increment < -36:
                            penalty = np.exp(np.log(df + 1) - _LOG2 + log_increment)
                        else:
                            penalty = 0.5 * (df + 1) * np.logaddexp(0, log_increment)
                        return float(np.exp(s - penalty))

                    integral = checked_integral(integrand)
                    result[index] = (
                        lz
                        + float(self.logpdf(lz, df))
                        - np.log(rate)
                        + np.log(integral)
                    )
        if np.any(np.isnan(result)):
            raise NotImplementedError("BCT tail exceeds the supported numerical range.")
        return result

    def _inverse_tail(self, log_p, tau):
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            quantile = stats.t.isf(np.exp(log_p), tau)
            result = np.array(np.log(quantile), copy=True)
        actual = self.logsf_abs(result, tau)
        unresolved = (
            (
                (~np.isfinite(result))
                | ~np.isclose(actual, log_p, rtol=2e-13, atol=2e-12)
            )
            & np.isfinite(log_p)
            & (log_p < _LOG_HALF)
        )
        for index in _indices(unresolved):
            target, df = float(log_p[index]), float(tau[index])

            def equation(lz, target=target, df=df):
                return float(self.logsf_abs(np.array([lz]), np.array([df]))[0] - target)

            lower, upper = -745.0, 1.0
            while equation(upper) > 0 and np.isfinite(upper):
                upper *= 2
            if not np.isfinite(upper):
                raise NotImplementedError(
                    "BCT inverse tail exceeds the supported numerical range."
                )
            result[index] = checked_root(equation, lower, upper)
        return result


class PowerExponentialLatent(_SymmetricLatent):
    @staticmethod
    def log_alpha(tau):
        return 0.5 * (special.gammaln(1 / tau) - special.gammaln(3 / tau))

    def validate(self, tau):
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            alpha = np.exp(self.log_alpha(tau))
        if np.any(~np.isfinite(alpha)) or np.any(alpha == 0):
            raise NotImplementedError(
                "BCPE standardized power-exponential scale is outside the supported float64 range."
            )

    def logpdf(self, log_abs, tau):
        alpha = self.log_alpha(tau)
        with np.errstate(over="ignore", under="ignore"):
            return (
                np.log(tau)
                - _LOG2
                - alpha
                - special.gammaln(1 / tau)
                - np.exp(tau * (log_abs - alpha))
            )

    def logsf_abs(self, log_abs, tau):
        log_abs, tau = np.broadcast_arrays(log_abs, tau)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            alpha = self.log_alpha(tau)
            log_x = tau * (log_abs - alpha)
            x = np.exp(log_x)
            log_q = _gamma_log_survival(1 / tau, x)
            # P(a,x)=x**a/Gamma(a+1)*(1+O(x)), with relative remainder
            # bounded by x. For log(x)<-36 that remainder is below float64
            # precision. Cancel a*tau analytically before x can underflow.
            small = log_x < -36
            log_p = log_abs - alpha - special.gammaln(1 + 1 / tau)
            log_q = np.where(small, log_complement(log_p), log_q)
            result = -_LOG2 + log_q
        if np.any(np.isnan(result)):
            raise NotImplementedError(
                "BCPE tail exceeds the supported numerical range."
            )
        return result

    def _inverse_tail(self, log_p, tau):
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            x = np.array(
                special.gammainccinv(1 / tau, np.exp(_LOG2 + log_p)), copy=True
            )
        unresolved = np.isinf(x) & np.isfinite(log_p)
        for index in _indices(unresolved):
            shape, target = 1 / float(tau[index]), _LOG2 + float(log_p[index])

            def equation(v, shape=shape, target=target):
                return float(
                    _gamma_log_survival(np.array([shape]), np.array([v]))[0] - target
                )

            upper = max(1.0, shape, -target)
            while equation(upper) > 0:
                upper *= 2
                if not np.isfinite(upper):
                    raise NotImplementedError(
                        "BCPE inverse tail exceeds the supported numerical range."
                    )
            x[index] = checked_root(equation, 0, upper)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = self.log_alpha(tau) + np.log(x) / tau
            # The gamma quantile can underflow while its 1/tau power remains
            # representable. Recover its log magnitude from the same small-x
            # identity, rather than rounding a nonmedian PE quantile to zero.
            small = (x == 0) & np.isfinite(log_p) & (log_p < _LOG_HALF)
            log_lower = log_complement(_LOG2 + log_p)
            result = np.where(
                small,
                self.log_alpha(tau) + log_lower + special.gammaln(1 + 1 / tau),
                result,
            )
            return result
