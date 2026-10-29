"""NumPy/SciPy probability laws with explicit natural parameterizations."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import stats

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
        "mean",
        "variance",
    }
)


def _lognormal_variance(loc, scale):
    """Evaluate lognormal variance without an inf-times-zero intermediate."""
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        s2 = scale**2
        correction = np.where(s2 > 1, s2 + np.log1p(-np.exp(-s2)), np.log(np.expm1(s2)))
        return np.exp(2 * loc + s2 + correction)


class _ScipyScalarLaw:
    """Shared operation dispatch; subclasses bind a validated probability law."""

    spec: ClassVar[DistributionSpec]
    positive_parameters: ClassVar[tuple[str, ...]]

    def _bind(self, parameters):
        raise NotImplementedError

    def _bind_context(self, parameters, context):
        return self._bind(parameters)

    def validate_parameters(self, parameters, *, context):
        for name in self.positive_parameters:
            if np.any(parameters[name] <= 0):
                raise ValueError(
                    f"{self.spec.name} parameter {name!r} must be positive."
                )

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        distribution = self._bind_context(parameters, context)
        if operation == "quantile":
            return distribution.ppf(values[:, None]).T
        if operation == "sample":
            return distribution.rvs(
                size=(n_draws, len(next(iter(parameters.values())))), random_state=rng
            )
        if operation == "cdf_left":
            return self._cdf_left(distribution, values)
        method = {
            "log_prob": "logpmf" if self.spec.response_kind == "discrete" else "logpdf",
            "cdf": "cdf",
            "cdf_left": "cdf",
            "log_cdf": "logcdf",
            "survival": "sf",
            "log_survival": "logsf",
            "mean": "mean",
            "variance": "var",
        }[operation]
        function = getattr(distribution, method)
        return function() if values is None else function(values)

    def _cdf_left(self, distribution, values):
        return distribution.cdf(values)


@dataclass(frozen=True)
class NormalLaw(_ScipyScalarLaw):
    """Normal probability law with real location ``loc`` and positive SD ``scale``."""

    spec = DistributionSpec(
        "normal", ("loc", "scale"), _OPERATIONS, "location_sd_v1", "continuous"
    )
    positive_parameters = ("scale",)

    def _bind(self, parameters):
        return stats.norm(loc=parameters["loc"], scale=parameters["scale"])


@dataclass(frozen=True)
class GammaLaw(_ScipyScalarLaw):
    """Gamma probability law with positive ``shape`` and inverse-scale ``rate``."""

    spec = DistributionSpec(
        "gamma", ("shape", "rate"), _OPERATIONS, "shape_rate_v1", "continuous"
    )
    positive_parameters = ("shape", "rate")

    def _bind(self, parameters):
        return stats.gamma(a=parameters["shape"], scale=1.0 / parameters["rate"])


@dataclass(frozen=True)
class LogNormalLaw(_ScipyScalarLaw):
    """Lognormal with real log-location loc and positive log-SD scale.

    Log-density and CDF operations work on log(y), avoiding exp(loc) overflow.
    Outcome quantiles, draws and moments retain ordinary float64 range limits.
    """

    spec = DistributionSpec(
        "lognormal", ("loc", "scale"), _OPERATIONS, "log_location_sd_v1", "continuous"
    )
    positive_parameters = ("scale",)

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        loc, scale = parameters["loc"], parameters["scale"]
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            if operation == "mean":
                return np.exp(loc + scale**2 / 2)
            if operation == "variance":
                return _lognormal_variance(loc, scale)
            if operation == "sample":
                return rng.lognormal(loc, scale, size=(n_draws, len(loc)))
            normal = stats.norm(loc=loc, scale=scale)
            if operation == "quantile":
                return np.exp(normal.ppf(values[:, None]).T)
            log_y = np.where(values <= 0, -np.inf, np.log(values))
            if operation == "log_prob":
                return np.where(values <= 0, -np.inf, normal.logpdf(log_y) - log_y)
            method = {
                "cdf": "cdf",
                "cdf_left": "cdf",
                "log_cdf": "logcdf",
                "survival": "sf",
                "log_survival": "logsf",
            }[operation]
            return getattr(normal, method)(log_y)


@dataclass(frozen=True)
class WeibullLaw(_ScipyScalarLaw):
    """Weibull probability law with positive scale and shape (WEI convention)."""

    spec = DistributionSpec(
        "weibull", ("scale", "shape"), _OPERATIONS, "scale_shape_v1", "continuous"
    )
    positive_parameters = ("scale", "shape")

    def _bind(self, parameters):
        return stats.weibull_min(c=parameters["shape"], scale=parameters["scale"])


@dataclass(frozen=True)
class StudentTLaw(_ScipyScalarLaw):
    """Student-t with positive df/scale and real location.

    Mean is NaN for df <= 1. Variance is NaN for df <= 1 and infinite
    for 1 < df <= 2. Probability operations remain available in these regimes.
    """

    spec = DistributionSpec(
        "studentt",
        ("df", "loc", "scale"),
        _OPERATIONS,
        "df_location_scale_v1",
        "continuous",
    )
    positive_parameters = ("df", "scale")

    def _bind(self, parameters):
        return stats.t(
            df=parameters["df"], loc=parameters["loc"], scale=parameters["scale"]
        )

    def evaluate(self, operation, parameters, **kwargs):
        df = parameters["df"]
        if operation == "mean":
            return np.where(df > 1, parameters["loc"], np.nan)
        if operation == "variance":
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = parameters["scale"] ** 2 * df / (df - 2)
            return np.where(df > 2, variance, np.where(df > 1, np.inf, np.nan))
        return super().evaluate(operation, parameters, **kwargs)


@dataclass(frozen=True)
class BetaLaw(_ScipyScalarLaw):
    """Beta law with positive alpha/beta shapes; queries do not clip responses."""

    spec = DistributionSpec(
        "beta", ("alpha", "beta"), _OPERATIONS, "two_shapes_v1", "continuous"
    )
    positive_parameters = ("alpha", "beta")

    def _bind(self, parameters):
        return stats.beta(a=parameters["alpha"], b=parameters["beta"])


@dataclass(frozen=True)
class NegativeBinomialLaw(_ScipyScalarLaw):
    """NB2 counts with mean mu>0, dispersion alpha>0 and Var=mu+alpha*mu**2.

    CDF queries accept real thresholds; left limits use ceil(y)-1. A zero
    quantile probability returns the lower support boundary zero. Parameter
    regimes whose SciPy success probability rounds to zero or one are rejected.
    """

    spec = DistributionSpec(
        "negativebinom",
        ("mean", "dispersion"),
        _OPERATIONS,
        "nb2_mean_dispersion_v1",
        "discrete",
    )
    positive_parameters = ("mean", "dispersion")

    def validate_parameters(self, parameters, *, context):
        super().validate_parameters(parameters, context=context)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            size = 1 / parameters["dispersion"]
            probability = 1 / (1 + parameters["dispersion"] * parameters["mean"])
        if np.any(~np.isfinite(size)) or np.any(
            (probability <= 0) | (probability >= 1)
        ):
            raise NotImplementedError(
                "Negative-binomial parameters exceed the representable SciPy size/probability range; no Poisson approximation is substituted."
            )

    def _bind(self, parameters):
        return stats.nbinom(
            n=1 / parameters["dispersion"],
            p=1 / (1 + parameters["dispersion"] * parameters["mean"]),
        )

    def _cdf_left(self, distribution, values):
        return distribution.cdf(np.ceil(values) - 1)

    def evaluate(self, operation, parameters, **kwargs):
        if operation == "mean":
            return parameters["mean"].copy()
        if operation == "variance":
            return (
                parameters["mean"] + parameters["dispersion"] * parameters["mean"] ** 2
            )
        result = super().evaluate(operation, parameters, **kwargs)
        if operation == "quantile":
            result[:, kwargs["values"] == 0] = 0
        return result


def validate_trials(trials, n_rows):
    """Validate explicit row-aligned binomial denominators; never recycle."""
    trials = np.asarray(trials, dtype=float)
    if (
        trials.shape != (n_rows,)
        or not np.all(np.isfinite(trials))
        or np.any(trials < 0)
        or np.any(trials != np.floor(trials))
    ):
        raise ValueError("trials must be a finite non-negative integer vector with one value per row.")
    return trials


@dataclass(frozen=True)
class BetaBinomialLaw(_ScipyScalarLaw):
    """Beta-binomial counts with mean probability mu, precision kappa and trials.

    Shapes are mu*kappa and (1-mu)*kappa. R BB uses sigma=1/kappa.
    No binomial approximation or dispersion clipping is substituted.
    """

    spec = DistributionSpec(
        "betabinomial", ("mu", "kappa"), _OPERATIONS,
        "mean_precision_trials_v1", "discrete", context_names=("trials",),
    )
    positive_parameters = ("kappa",)

    def validate_parameters(self, parameters, *, context):
        super().validate_parameters(parameters, context=context)
        mu, kappa = parameters["mu"], parameters["kappa"]
        if np.any((mu <= 0) | (mu >= 1)):
            raise ValueError("Beta-binomial mu must be inside (0,1).")
        validate_trials(context["trials"], len(mu))
        if np.any(mu * kappa <= 0) or np.any((1 - mu) * kappa <= 0):
            raise NotImplementedError("Beta-binomial shapes underflow; no limiting approximation is substituted.")

    def _bind_context(self, parameters, context):
        mu, kappa = parameters["mu"], parameters["kappa"]
        return stats.betabinom(context["trials"], mu * kappa, (1 - mu) * kappa)

    def _cdf_left(self, distribution, values):
        return distribution.cdf(np.ceil(values) - 1)

    def evaluate(self, operation, parameters, *, context, **kwargs):
        mu, kappa = parameters["mu"], parameters["kappa"]
        trials = context["trials"]
        if operation == "mean":
            return trials * mu
        if operation == "variance":
            return trials * mu * (1 - mu) * (trials + kappa) / (1 + kappa)
        result = super().evaluate(operation, parameters, context=context, **kwargs)
        if operation == "quantile":
            result[:, kwargs["values"] == 0] = 0
        return result
