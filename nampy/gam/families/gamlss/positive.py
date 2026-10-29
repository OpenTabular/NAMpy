"""Positive-response LSS extensions with exact joint likelihood derivatives.

Probability parameterizations follow gamlss.dist::LOGNO (R/logNO.R) and
gamlss.dist::WEI (R/WEI.R). Fitting uses NAMpy's existing mgcv joint engine,
not GAMLSS backfitting or its expected/working curvature approximations.
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import weibull_min

from ....distributions.laws import _lognormal_variance
from .distributional import _DistributionalFamily


class _PositiveDistributionalFamily(_DistributionalFamily):
    def validate_y(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError(
                f"{self.name} requires finite, strictly positive responses."
            )
        return y


class LogNormalLocationScale(_PositiveDistributionalFamily):
    """Lognormal LSS: log(Y) ~ Normal(loc, scale), with scale>0.

    No constructor options. Predictor keys and natural columns are (loc,scale),
    with identity/log links. loc is unrestricted log-location, not the response
    mean; E(Y)=exp(loc+scale**2/2). Matches LOGNO(mu=loc,sigma=scale), not LOGNO2
    or LNO. The likelihood retains the response Jacobian -log(y). Exact
    likelihood derivatives through order four support the existing general
    REML route. Other method labels follow mgcv's general-family coercion to
    REML; no separate GCV/UBRE/NCV criterion or censored likelihood is provided.
    """

    name = "lognormal_lss"
    supports_ml = supports_reml = supports_laml = True
    parameter_names = ("loc", "scale")
    link_names = ("identity", "log")
    parameter_bounds = {"scale": (0, None)}
    max_derivative_order = 4

    def _log_density(self, y, loc, log_scale):
        log_y = np.log(y)
        z = (log_y - loc) * (-log_scale).exp()
        return -0.5 * np.log(2 * np.pi) - log_scale - 0.5 * z * z - log_y

    def _initial_predictors(self, y):
        log_y = np.log(y)
        return log_y, np.log(max(np.std(log_y), 1e-3))

    def logpdf_from_parameters(self, y, parameters):
        loc, scale = self._check_parameters(parameters).T
        log_y = np.log(self.validate_y(y))
        return (
            -0.5 * np.log(2 * np.pi)
            - np.log(scale)
            - 0.5 * ((log_y - loc) / scale) ** 2
            - log_y
        )

    def point_prediction_from_parameters(self, parameters):
        loc, scale = self._check_parameters(parameters).T
        return np.exp(loc + 0.5 * scale**2)

    def variance_from_parameters(self, parameters):
        loc, scale = self._check_parameters(parameters).T
        return _lognormal_variance(loc, scale)

    def rd(self, mu, wt, scale, *, rng):
        loc, sd = self._check_parameters(mu).T
        return rng.lognormal(loc, sd)


class WeibullLocationShape(_PositiveDistributionalFamily):
    """Weibull LSS with positive natural columns (scale,shape), both log-linked.

    No constructor options. Matches WEI(mu=scale,sigma=shape), not WEI2/WEI3.
    The conditional mean is scale*Gamma(1+1/shape), not its first parameter.
    Exact likelihood derivatives through order four support the existing joint
    REML route. Other method labels follow mgcv general-family coercion to
    REML. Censoring, truncation, alternate links, separate GCV/UBRE and general
    NCV fitting are not implemented by this family.
    """

    name = "weibull_lss"
    supports_ml = supports_reml = supports_laml = True
    parameter_names = ("scale", "shape")
    link_names = ("log", "log")
    parameter_bounds = {"scale": (0, None), "shape": (0, None)}
    max_derivative_order = 4

    def _log_density(self, y, log_scale, log_shape):
        log_y = np.log(y)
        t = log_shape.exp() * (log_y - log_scale)
        return log_shape - log_y + t - t.exp()

    def _initial_predictors(self, y):
        log_y = np.log(y)
        shape = np.pi / (np.sqrt(6) * max(np.std(log_y), 1e-3))
        return log_y + np.euler_gamma / shape, np.log(shape)

    def logpdf_from_parameters(self, y, parameters):
        scale, shape = self._check_parameters(parameters).T
        log_y = np.log(self.validate_y(y))
        t = shape * (log_y - np.log(scale))
        return np.log(shape) - log_y + t - np.exp(t)

    def point_prediction_from_parameters(self, parameters):
        scale, shape = self._check_parameters(parameters).T
        return np.exp(np.log(scale) + gammaln(1 + 1 / shape))

    def variance_from_parameters(self, parameters):
        scale, shape = self._check_parameters(parameters).T
        return weibull_min.var(c=shape, scale=scale)

    def rd(self, mu, wt, scale, *, rng):
        value, shape = self._check_parameters(mu).T
        return value * rng.weibull(shape)
