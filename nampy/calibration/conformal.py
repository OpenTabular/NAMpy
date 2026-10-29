"""Split conformal regression with explicit finite-sample order statistics.

Romano, Patterson and Candès (2019), equations 4--6. The calibration data
must be exchangeable with prediction observations, independently of model
fitting/tuning. Scores are never interpolated, clipped or silently dropped. Optional density
ratios implement covariate-shift calibration, not likelihood case weights.
"""

from dataclasses import dataclass
from math import ceil
from numbers import Real

import numpy as np

from ..distributions import DistributionPrediction


def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real scalar responses.")
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector of scalar responses.")
    return array


def _level(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("level must lie strictly between 0 and 1.")
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError("level must lie strictly between 0 and 1.")
    return result


def _bounds(prediction, method, level):
    if isinstance(prediction, DistributionPrediction):
        prediction = (prediction.mean() if method == "absolute" else
                      prediction.quantile([(1-level)/2, (1+level)/2]))
    if method == "absolute":
        point = _vector(prediction, "prediction")
        return point, point
    if np.iscomplexobj(prediction):
        raise ValueError("Quantile predictions must be real.")
    array = np.array(prediction, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[1] != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Quantile predictions must be finite (n,2) lower/upper bounds.")
    if np.any(array[:, 0] > array[:, 1]):
        raise ValueError("Quantile predictions cross; automatic sorting or repair is unsupported.")
    return array[:, 0], array[:, 1]


@dataclass(frozen=True)
class ConformalInterval:
    """Future scalar-response prediction sets; lower > upper denotes an empty set.

    Marginal coverage under exchangeability is distinct from coefficient or
    conditional-quantile inference. Infinite limits are intentional when the
    finite-sample calibration rank exceeds the calibration sample size.
    """

    lower: np.ndarray
    upper: np.ndarray
    level: float
    method: str
    n_calibration: int
    correction: float | np.ndarray
    coverage: str = "marginal_under_exchangeability"
    uncertainty_kind: str = "future_observation_conformal"
    scope: str = "one_observation"
    calibration_counts: np.ndarray | None = None
    weight_source: str | None = None

    def __post_init__(self):
        for name in ("lower", "upper", "correction", "calibration_counts"):
            value = getattr(self, name)
            if value is None or (name == "correction" and np.ndim(value) == 0):
                continue
            array = np.array(value, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def empty(self):
        return self.lower > self.upper


@dataclass(frozen=True)
class ConformalCalibration:
    """Immutable scores for one method/coverage level and an unchanged predictor.

    Array-level callers own the predictor association. Prefer calibrate_regressor
    for estimator use: it snapshots the fitted model before scoring. Quantile
    scores may be negative; their correction can shrink or empty intervals.
    """

    scores: np.ndarray
    method: str = "absolute"
    level: float = .9
    density_ratio: np.ndarray | None = None
    weight_source: str = "known"

    def __post_init__(self):
        if self.method not in {"absolute", "quantile"}:
            raise NotImplementedError("Only absolute-residual and quantile split conformal are supported.")
        if self.weight_source not in {"known", "estimated"}:
            raise ValueError("weight_source must be known or estimated.")
        if self.density_ratio is None and self.weight_source != "known":
            raise ValueError("weight_source requires density ratios.")
        if self.density_ratio is not None:
            from .weighted import density_ratios
            ratios = density_ratios(self.density_ratio)
            if len(ratios) != len(self.scores):
                raise ValueError("Calibration density ratios must align with scores.")
            object.__setattr__(self, "density_ratio", ratios)
        scores = _vector(self.scores, "scores")
        if self.method == "absolute" and np.any(scores < 0):
            raise ValueError("Absolute-residual scores must be nonnegative.")
        scores.setflags(write=False)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "level", _level(self.level))

    def __reduce__(self):
        # Re-run validation/read-only normalization when restoring fitted state.
        return type(self), (self.scores, self.method, self.level, self.density_ratio, self.weight_source)

    @property
    def rank(self):
        """One-based ceil((n+1)*level), without percentile interpolation."""
        if self.density_ratio is not None:
            raise ValueError("Weighted quantiles require per-prediction density ratios, not one global rank.")
        return ceil((len(self.scores)+1)*self.level)

    @property
    def correction(self):
        if self.rank > len(self.scores):
            return np.inf
        return float(np.partition(self.scores, self.rank-1)[self.rank-1])

    def corrections(self, density_ratio=None):
        """Scalar unweighted correction or one weighted correction per test row."""
        if self.density_ratio is None:
            if density_ratio is not None:
                raise ValueError("Test density ratios require weighted calibration.")
            return self.correction
        if density_ratio is None:
            raise ValueError("Weighted calibration requires explicit test-point density_ratio.")
        from .weighted import weighted_corrections
        return weighted_corrections(self.scores, self.density_ratio, density_ratio, self.level)

    def interval(self, prediction, *, density_ratio=None):
        """Evaluate new predictions from the same fixed model used for scoring."""
        lower, upper = _bounds(prediction, self.method, self.level)
        correction = self.corrections(density_ratio)
        if np.ndim(correction) and np.shape(correction) != lower.shape:
            raise ValueError("Test density ratios must align with prediction rows.")
        with np.errstate(over="raise", invalid="raise"):
            lower, upper = lower-correction, upper+correction
        coverage = ("marginal_under_exchangeability" if self.density_ratio is None else
                    "marginal_under_known_covariate_shift" if self.weight_source == "known" else
                    "approximate_with_estimated_density_ratio")
        return ConformalInterval(lower, upper, self.level, self.method,
                                 len(self.scores), correction, coverage=coverage,
                                 weight_source=None if self.density_ratio is None else self.weight_source)


def calibrate_conformal(y, prediction, *, method="absolute", level=.9,
                        density_ratio=None, weight_source="known"):
    """Calibrate held-out scalar responses against predictions or a distribution.

    Absolute mode uses point predictions (the mean for DistributionPrediction).
    Quantile mode accepts (n,2) bounds or distribution quantiles at the central
    level. Unsupported moments/probability operations propagate explicitly.
    No rows are dropped. Censoring targets and vector responses are unsupported.
    """
    level = _level(level)
    if method not in {"absolute", "quantile"}:
        raise NotImplementedError("Only absolute-residual and quantile split conformal are supported.")
    y = _vector(y, "y")
    lower, upper = _bounds(prediction, method, level)
    if y.shape != lower.shape:
        raise ValueError("Responses and predictions must have identical row counts.")
    with np.errstate(over="raise", invalid="raise"):
        scores = np.maximum(lower-y, y-upper)
    return ConformalCalibration(scores, method, level, density_ratio, weight_source)
