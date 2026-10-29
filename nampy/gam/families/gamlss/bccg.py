"""BCCG joint likelihood family with analytic REML smoothing derivatives.

The normalized density follows gamlss.dist/R/BCCG.R::dBCCG. Its d2ld* hooks
are working/expected curvature and are not the observed joint Hessian used
by mgcv's general-family engine (R/gamlss.r::gamlss.gH).
"""

from math import factorial

import numpy as np
from scipy.special import log_ndtr

from ....distributions import BCCGLaw, DistributionPrediction
from .positive import _PositiveDistributionalFamily


def _log_normalizer(t):
    """Compose log Phi(1/abs(t)) through fourth order, flat at t=0.

    Rescale the Taylor increment before composing the inverse Mills ratio.
    A tiny unary derivative can otherwise underflow before multiplication by
    a large predictor Jacobian, losing a representable mixed derivative.
    """
    if t.order > 4:
        raise NotImplementedError(
            "BCCG normalizer derivatives are available only through order four."
        )
    x = t.value
    if any(np.any(~np.isfinite(c)) for c in t.coefficients.values()):
        raise NotImplementedError(
            "BCCG normalizer predictors exceed the supported floating-point range."
        )
    value = np.zeros_like(x)
    scale = np.ones_like(x)
    for index, coefficient in t.coefficients.items():
        if sum(index):
            scale = np.maximum(scale, np.abs(coefficient))
    nonzero = x != 0
    a = np.zeros_like(x)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        a[nonzero] = 1 / np.abs(x[nonzero])
        # For a>=100 the Gaussian exponent dominates a**11 times even four
        # maximal finite chain factors: -5000+11*log(100)+4*log(float_max)
        # is below log(smallest_subnormal). All derivatives round to zero.
        active = nonzero & (a < 100)
        aa = a[active]
        value[active] = log_ndtr(aa)
        log_r = -0.5 * aa * aa - 0.5 * np.log(2 * np.pi) - value[active]
        log_scale = np.log(scale[active])
        r = np.exp(log_r)
        # d^k/dt^k log Phi(a) = sign(t)^k * r*a^(k+1)*P_k(a,r),
        # using da/dt=-sign(t)*a^2 and dr/da=-r*(a+r).
        polynomials = [
            -np.ones_like(aa),
            2 - aa**2 - aa * r,
            -6 + 7 * aa**2 - aa**4 + (6 * aa - 3 * aa**3) * r - 2 * aa**2 * r**2,
            24
            - 48 * aa**2
            + 15 * aa**4
            - aa**6
            + (-36 * aa + 40 * aa**3 - 7 * aa**5) * r
            + (24 * aa**2 - 12 * aa**4) * r**2
            - 6 * aa**3 * r**3,
        ]
        derivatives = []
        for degree in range(1, t.order + 1):
            p = polynomials[degree - 1]
            derivative = np.zeros_like(x)
            derivative[active] = (
                np.sign(x[active]) ** degree
                * np.sign(p)
                * np.exp(
                    log_r
                    + (degree + 1) * np.log(aa)
                    + np.log(np.abs(p))
                    + degree * log_scale
                )
            )
            derivatives.append(derivative)
        delta = (t - t.value) / scale
        result, power = t.constant(value), t.constant(1)
        for degree, derivative in enumerate(derivatives, 1):
            power = power * delta
            result = result + power * (derivative / factorial(degree))
    return result


class BCCGLikelihood(_PositiveDistributionalFamily):
    """Three-predictor BCCG family, with log/log/identity links.

    Observed likelihood derivatives through order four use the generic
    extension adapter for coefficient and outer derivative assembly.
    Initialization follows BCCG.R, projected by the generic adapter. Fixed
    smoothing and joint REML with Newton, BFGS and EFS are supported. Other
    method labels retain general-family REML coercion. Neural fitting,
    alternate links, censoring/truncation and optim/magic/nlm.fd are unsupported.
    This kernel does not implement GAMLSS RS/backfitting or mgcv censored bcg.
    """

    name = "bccg_lss"
    parameter_names = ("mu", "sigma", "nu")
    link_names = ("log", "log", "identity")
    parameter_bounds = {"mu": (0, None), "sigma": (0, None)}
    max_derivative_order = 4
    supports_ml = supports_reml = supports_laml = True

    def _log_density(self, y, log_mu, log_sigma, nu):
        if not np.all(
            np.isfinite(np.column_stack([log_mu.value, log_sigma.value, nu.value]))
        ):
            raise ValueError("BCCG predictors must be finite.")
        log_ratio = np.log(y) - log_mu
        sigma = log_sigma.exp()
        z = log_ratio * (-log_sigma).exp() * (nu * log_ratio).expm1_over_x()
        value = (
            nu * log_ratio
            - log_sigma
            - 0.5 * z * z
            - np.log(y)
            - 0.5 * np.log(2 * np.pi)
            - _log_normalizer(sigma * nu)
        )
        if any(np.any(~np.isfinite(c)) for c in value.coefficients.values()):
            raise NotImplementedError(
                "BCCG likelihood derivatives exceed the supported floating-point range."
            )
        return value

    def _initial_predictors(self, y):
        # gamlss.dist/R/BCCG.R: mu=(y+mean(y))/2, sigma=.1, nu=.5.
        return np.log((y + np.mean(y)) / 2), np.log(0.1), 0.5

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y = self.validate_y(y)
        active = (
            np.ones(len(y), dtype=bool) if weights is None else np.asarray(weights) > 0
        )
        if len(np.unique(y[active])) < 3:
            raise ValueError(
                "BCCG fitting requires at least three distinct positive-weight responses; degenerate location/scale/power fits are unsupported."
            )
        return super().initialize(y, X, jj, offset=offset, weights=weights, E=E)

    def _distribution(self, parameters):
        parameters = self._check_parameters(parameters)
        return DistributionPrediction(
            BCCGLaw(), dict(zip(self.parameter_names, parameters.T, strict=True))
        )

    def logpdf_from_parameters(self, y, parameters):
        return self._distribution(parameters).log_prob(self.validate_y(y))

    def point_prediction_from_parameters(self, parameters):
        return self._distribution(parameters).mean()

    def variance_from_parameters(self, parameters):
        return self._distribution(parameters).variance()

    def rd(self, mu, wt, scale, *, rng):
        return self._distribution(mu).sample(1, random_state=rng)[0]
