"""Pointwise inference for calibrated conditional-quantile functions."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class QuantileFunctionInterval:
    estimate: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    quantile_level: float
    uncertainty_kind: str = "calFast_calibrated_quantile_function"
    coverage: str = "approximate_pointwise_fixed_smoothing"

    def __post_init__(self):
        for name in ("estimate", "standard_error", "lower", "upper"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def quantile_function_interval(estimate, matrix, covariance, *, level, quantile_level):
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie in (0,1).")
    variance = np.einsum("np,pq,nq->n", matrix, covariance, matrix)
    if np.any(variance < 0) or not np.isfinite(variance).all():
        raise FloatingPointError("Quantile-function variance is invalid.")
    se = np.sqrt(variance)
    width = norm.ppf((1 + level) / 2) * se
    return QuantileFunctionInterval(
        estimate, se, estimate - width, estimate + width, level, quantile_level
    )
