"""Row-aligned conditional response distributions shared by both backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .contracts import DistributionLaw, DistributionSpec


@dataclass(frozen=True)
class DistributionPrediction:
    """Snapshot of conditional distributions at specified prediction rows.

    Parameters
    ----------
    law : DistributionLaw
        Probability implementation and its explicit operation capabilities.
    parameters : mapping of str to ndarray
        One vector of shape ``(n,)`` per named natural parameter. Arrays are
        copied and made read-only. All-NaN rows preserve missing predictions;
        partial-NaN rows and infinite parameters are rejected.
    context : mapping of str to ndarray, optional
        Fixed rowwise inputs declared by the law, for example trial counts.
        Each vector has shape ``(n,)`` and is copied, made read-only and sliced
        with the parameter mask. The law validates its context domains.

    Notes
    -----
    Probability evaluations and moments return ``(n,)``. Quantiles return
    ``(n, n_probs)``, including ``(n, 1)`` for a scalar probability. Samples
    return ``(n_draws, n)``. Samples condition on these parameters and do not
    include coefficient or training uncertainty. Vector events are not yet
    supported. No estimator or fitting state is retained.
    """

    law: DistributionLaw
    parameters: Mapping[str, np.ndarray]
    context: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = self.law.spec
        if spec.event_shape:
            raise NotImplementedError(
                "DistributionPrediction currently supports scalar events only."
            )
        if set(self.parameters) != set(spec.parameter_names):
            raise ValueError(
                f"{spec.name} requires exactly the parameters {spec.parameter_names}."
            )
        parameters = {}
        n_rows = None
        for name in spec.parameter_names:
            value = np.array(self.parameters[name], dtype=np.float64, copy=True)
            if value.ndim != 1 or (n_rows is not None and len(value) != n_rows):
                raise ValueError(
                    "Distribution parameters must be equal-length vectors of shape (n,)."
                )
            n_rows = len(value)
            value.setflags(write=False)
            parameters[name] = value
        matrix = np.column_stack(list(parameters.values()))
        missing = np.all(np.isnan(matrix), axis=1)
        if np.any(~np.isfinite(matrix[~missing])):
            raise ValueError(
                "Parameters must be finite or all NaN in a missing prediction row."
            )
        if set(self.context) != set(spec.context_names):
            raise ValueError(
                f"{spec.name} requires exactly the context {spec.context_names}."
            )
        context = {}
        for name in spec.context_names:
            value = np.array(self.context[name], copy=True)
            if value.shape != (n_rows,):
                raise ValueError(f"Context {name!r} must have shape ({n_rows},).")
            value.setflags(write=False)
            context[name] = value
        self.law.validate_parameters(
            {name: value[~missing] for name, value in parameters.items()},
            context={name: value[~missing] for name, value in context.items()},
        )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "context", MappingProxyType(context))

    def __reduce__(self):
        # Reconstruct through validation so snapshots also remain read-only
        # after pickling; MappingProxyType itself cannot be pickled.
        return type(self), (self.law, dict(self.parameters), dict(self.context))

    @property
    def spec(self) -> DistributionSpec:
        """Law identity, parameterization, and probability capabilities."""
        return self.law.spec

    @property
    def n_rows(self) -> int:
        """Number of prediction rows, including missing rows."""
        return len(self.parameters[self.spec.parameter_names[0]])

    def _evaluate(
        self, operation, *, values=None, n_draws=None, rng=None
    ) -> np.ndarray:
        if operation not in self.spec.operations:
            raise NotImplementedError(
                f"{self.spec.name} does not support {operation!r}. "
                f"Available operations: {', '.join(sorted(self.spec.operations)) or 'none'}."
            )
        valid = ~np.isnan(self.parameters[self.spec.parameter_names[0]])
        shape: tuple[int, ...]
        expected: tuple[int, ...]
        if operation == "quantile":
            shape = (self.n_rows, len(values))
            expected = (int(valid.sum()), len(values))
        elif operation == "sample":
            shape = (n_draws, self.n_rows)
            expected = (n_draws, int(valid.sum()))
        else:
            shape = (self.n_rows,)
            expected = (int(valid.sum()),)
        result = np.full(shape, np.nan)
        if not np.any(valid):
            return result
        evaluated = np.asarray(
            self.law.evaluate(
                operation,
                {name: parameter[valid] for name, parameter in self.parameters.items()},
                context={name: value[valid] for name, value in self.context.items()},
                values=values
                if operation == "quantile" or values is None
                else values[valid],
                n_draws=n_draws,
                rng=rng,
            ),
            dtype=np.float64,
        )
        if evaluated.shape != expected:
            raise ValueError(
                f"{self.spec.name}.{operation} returned shape {evaluated.shape}; expected {expected}."
            )
        if operation == "sample":
            result[:, valid] = evaluated
        else:
            result[valid] = evaluated
        return result

    def _at(self, operation, y) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.n_rows, values.item())
        elif values.shape != (self.n_rows,):
            raise ValueError(f"y must be a scalar or have shape ({self.n_rows},).")
        return self._evaluate(operation, values=values)

    def log_prob(self, y) -> np.ndarray:
        """Unweighted normalized log density or log mass at scalar/rowwise y."""
        return self._at("log_prob", y)

    def cdf(self, y) -> np.ndarray:
        """Return P(Y <= y) for scalar or rowwise y."""
        return self._at("cdf", y)

    def cdf_left(self, y) -> np.ndarray:
        """Return P(Y < y), including the left limit at atoms."""
        return self._at("cdf_left", y)

    def log_cdf(self, y) -> np.ndarray:
        """Return the law's log-CDF operation without taking log(cdf(y))."""
        return self._at("log_cdf", y)

    def survival(self, y) -> np.ndarray:
        """Return P(Y > y)."""
        return self._at("survival", y)

    def log_survival(self, y) -> np.ndarray:
        """Return the law's log-survival operation."""
        return self._at("log_survival", y)

    def quantile(self, probs) -> np.ndarray:
        """Quantiles at finite probabilities in [0, 1], shape (n, n_probs)."""
        values = np.asarray(probs, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if (
            values.ndim != 1
            or np.any(~np.isfinite(values))
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError(
                "probs must be a scalar or vector of finite probabilities in [0, 1]."
            )
        return self._evaluate("quantile", values=values)

    def sample(self, n_draws=1, *, random_state=None) -> np.ndarray:
        """Draw conditional outcomes using an integer seed or NumPy Generator."""
        if (
            isinstance(n_draws, bool)
            or not isinstance(n_draws, Integral)
            or n_draws < 0
        ):
            raise ValueError("n_draws must be a nonnegative integer.")
        return self._evaluate(
            "sample", n_draws=int(n_draws), rng=np.random.default_rng(random_state)
        )

    def mean(self) -> np.ndarray:
        """Return conditional means, where declared by the law."""
        return self._evaluate("mean")

    def variance(self) -> np.ndarray:
        """Return conditional variances, where declared by the law."""
        return self._evaluate("variance")
