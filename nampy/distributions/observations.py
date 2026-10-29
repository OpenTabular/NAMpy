"""Explicit interval observations and fixed-bound truncation of scalar laws."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .contracts import DistributionLaw, DistributionSpec
from .laws import NormalLaw


@dataclass(frozen=True, eq=False)
class ObservationTarget:
    """Exact, left/right-censored or interval observations [lower, upper].

    Equal finite endpoints denote exact observations. Infinite endpoints denote
    censoring, never exact values. The finite encoding exists only for batching;
    these bounds are observation metadata and never sample weights.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lower, upper = (
            np.array(v, dtype=float, copy=True) for v in (self.lower, self.upper)
        )
        if (
            lower.ndim != 1
            or upper.shape != lower.shape
            or not len(lower)
            or np.isnan(lower).any()
            or np.isnan(upper).any()
            or np.any(lower > upper)
            or np.isposinf(lower).any()
            or np.isneginf(upper).any()
        ):
            raise ValueError(
                "Observation endpoints must be nonempty aligned ordered vectors; exact values must be finite."
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def __len__(self):
        return len(self.lower)

    def subset(self, index):
        return type(self)(
            np.atleast_1d(self.lower[index]), np.atleast_1d(self.upper[index])
        )

    @property
    def encoded(self):
        lo, hi = self.lower, self.upper
        code = np.where(
            lo == hi, 0, np.where(np.isneginf(lo), -1, np.where(np.isposinf(hi), 1, 2))
        )
        # Full uninformative intervals use a separate explicit code.
        code = np.where(np.isneginf(lo) & np.isposinf(hi), 3, code)
        value = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0))
        upper = np.where(np.isfinite(hi), hi, value)
        return np.column_stack([value, upper, code])

    @classmethod
    def from_encoded(cls, value):
        value = np.asarray(value, dtype=float)
        if (
            value.ndim != 2
            or value.shape[1] != 3
            or not np.isfinite(value).all()
            or not np.isin(value[:, 2], [-1, 0, 1, 2, 3]).all()
        ):
            raise ValueError(
                "Encoded observations require finite (value, upper, code) rows."
            )
        y, upper, code = value.T
        if np.any((code == 0) & (y != upper)) or np.any((code == 2) & (y >= upper)):
            raise ValueError("Encoded exact/interval endpoints are inconsistent.")
        return cls(
            np.where(np.isin(code, [-1, 3]), -np.inf, y),
            np.where(np.isin(code, [1, 3]), np.inf, upper),
        )

    def __reduce__(self):
        return type(self), (self.lower, self.upper)


def log_interval_mass(law, parameters, context, lower, upper):
    """Stable log P(lower<Y<=upper), using the numerically separated tail."""
    lower, upper = np.broadcast_arrays(lower, upper)
    if lower.ndim != 2 or lower.shape[0] != len(next(iter(parameters.values()))):
        raise ValueError("Interval queries require shape (n, m).")
    output = []
    for column in range(lower.shape[1]):
        lo, hi = lower[:, column], upper[:, column]
        a = law.evaluate("log_cdf", parameters, context=context, values=lo)
        b = law.evaluate("log_cdf", parameters, context=context, values=hi)
        sa = law.evaluate("log_survival", parameters, context=context, values=lo)
        sb = law.evaluate("log_survival", parameters, context=context, values=hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = b + np.log(-np.expm1(a - b))
            sf = sa + np.log(-np.expm1(sb - sa))
        output.append(np.where(lo >= hi, -np.inf, np.where(b <= np.log(0.5), cdf, sf)))
    return np.column_stack(output)


def observation_log_prob(prediction, target):
    """Log density for exact observations, log mass for censor intervals."""
    if not isinstance(target, ObservationTarget) or len(target) != prediction.n_rows:
        raise ValueError("An aligned ObservationTarget is required.")
    if prediction.spec.response_kind != "continuous":
        raise NotImplementedError(
            "Censoring currently requires scalar continuous response laws."
        )
    exact = target.lower == target.upper
    values = target.encoded[:, 0]
    density = prediction.log_prob(values)
    mass = log_interval_mass(
        prediction.law,
        prediction.parameters,
        prediction.context,
        target.lower[:, None],
        target.upper[:, None],
    )[:, 0]
    return np.where(exact, density, mass)


@dataclass(frozen=True)
class TruncatedLaw:
    """Condition a continuous scalar law on fixed (lower, upper) bounds.

    Probability capabilities derive from the base. Moments currently require a
    normal base. Generic inverse-CDF cases that lose tail probability precision
    raise; normal quantiles use SciPy's tail-stable truncated-normal inverse.
    """

    base: DistributionLaw
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self):
        if np.isnan(self.lower) or np.isnan(self.upper) or not self.lower < self.upper:
            raise ValueError("Truncation requires ordered, non-NaN fixed bounds.")
        if self.base.spec.event_shape or self.base.spec.response_kind != "continuous":
            raise NotImplementedError(
                "Truncation currently requires a scalar continuous base law."
            )
        required = {"log_prob", "log_cdf", "log_survival"}
        if not required <= self.base.spec.operations:
            raise NotImplementedError(
                "Truncation requires base density and both log tails."
            )

    @property
    def spec(self):
        operations = {
            "log_prob",
            "cdf",
            "cdf_left",
            "log_cdf",
            "survival",
            "log_survival",
        }
        if "quantile" in self.base.spec.operations:
            operations |= {"quantile", "sample"}
        if type(self.base) is NormalLaw:
            operations |= {"mean", "variance"}
        return DistributionSpec(
            "truncated_" + self.base.spec.name,
            self.base.spec.parameter_names,
            frozenset(operations),
            f"fixed_truncation_v1:{self.lower!r}:{self.upper!r}:{self.base.spec.parameterization}",
            "continuous",
            context_names=self.base.spec.context_names,
        )

    def validate_parameters(self, parameters, *, context):
        self.base.validate_parameters(parameters, context=context)
        n = len(next(iter(parameters.values())))
        normalizer = log_interval_mass(
            self.base,
            parameters,
            context,
            np.full((n, 1), self.lower),
            np.full((n, 1), self.upper),
        )
        if not np.isfinite(normalizer).all():
            raise NotImplementedError(
                "Truncation mass is outside the supported numerical range."
            )

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        if operation not in self.spec.operations:
            raise NotImplementedError(f"{self.spec.name} does not support {operation}.")
        n = len(next(iter(parameters.values())))
        lo, hi = np.full((n, 1), self.lower), np.full((n, 1), self.upper)
        log_z = log_interval_mass(self.base, parameters, context, lo, hi)
        if (
            operation in {"mean", "variance", "quantile", "sample"}
            and type(self.base) is NormalLaw
        ):
            mean, scale = parameters["loc"], parameters["scale"]
            bound = stats.truncnorm(
                (self.lower - mean) / scale,
                (self.upper - mean) / scale,
                loc=mean,
                scale=scale,
            )
            if operation == "mean":
                return bound.mean()
            if operation == "variance":
                return bound.var()
            probabilities = (
                rng.random((n_draws, n)).T
                if operation == "sample"
                else np.broadcast_to(values, (n, len(values)))
            )
            answer = bound.ppf(probabilities.T).T
            return answer.T if operation == "sample" else answer
        if operation in {"quantile", "sample"}:
            p = (
                rng.random((n_draws, n)).T
                if operation == "sample"
                else np.broadcast_to(values, (n, len(values)))
            )
            cdf_lo = np.exp(
                self.base.evaluate("log_cdf", parameters, context=context, values=lo[:, 0])
            )[:, None]
            probability = cdf_lo + p * np.exp(log_z)
            if np.any((p > 0) & (p < 1) & ((probability <= 0) | (probability >= 1))):
                raise NotImplementedError(
                    "Generic truncated inverse CDF loses tail precision; this range is unsupported."
                )
            result = np.empty_like(p)
            for i in range(n):
                row = {k: v[i : i + 1] for k, v in parameters.items()}
                ctx = {k: v[i : i + 1] for k, v in context.items()}
                result[i] = self.base.evaluate(
                    "quantile", row, context=ctx, values=probability[i]
                )[0]
            support = self.base.evaluate(
                "quantile", parameters, context=context, values=np.array([0.0, 1.0])
            )
            result = np.where(
                p == 0, np.maximum(self.lower, support[:, 0])[:, None], result
            )
            result = np.where(
                p == 1, np.minimum(self.upper, support[:, 1])[:, None], result
            )
            return result.T if operation == "sample" else result
        if operation == "log_prob":
            density = (
                self.base.evaluate(
                    operation, parameters, context=context, values=values
                )
                - log_z[:, 0]
            )
            return np.where(
                (values > self.lower) & (values < self.upper), density, -np.inf
            )
        x = np.clip(values, self.lower, self.upper)[:, None]
        result = (
            log_interval_mass(self.base, parameters, context, lo, x)
            if operation in {"cdf", "cdf_left", "log_cdf"}
            else log_interval_mass(self.base, parameters, context, x, hi)
        ) - log_z
        result = result[:, 0]
        return result if operation in {"log_cdf", "log_survival"} else np.exp(result)
