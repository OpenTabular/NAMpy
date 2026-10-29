"""Censor-aware one-sided bounds, with explicit conservative/cutoff assumptions.

The cutoff route follows Candès, Lei and Ren, Conformalized Survival Analysis,
Algorithm 1, with score min(q_alpha(X),cutoff)-min(T,cutoff). Censoring times
must be observed for *all* records; ordinary event indicators cannot supply them.

References
----------
- Candès, E. J., Lei, L., and Ren, Z. (2023). Conformalized Survival Analysis.
  https://arxiv.org/abs/2103.09763
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..persistence import load_estimator, save_estimator, snapshot_estimator
from ..survival import SurvivalPrediction, SurvivalTarget
from .conformal import ConformalCalibration, ConformalInterval, _level, _vector


def _base(prediction, level, cutoff):
    if isinstance(prediction, SurvivalPrediction):
        base = prediction.quantile(1-level)[:, 0]
        if cutoff is not None:
            # A quantile beyond finite piecewise support can still be capped
            # exactly when S(cutoff)>level. This is not tail extrapolation.
            beyond = prediction.survival([cutoff])[:, 0] > level
            base = np.where(beyond, cutoff, base)
    else:
        base = prediction
    base = _vector(base, "survival lower-quantile prediction")
    if np.any(base < 0):
        raise ValueError("Survival base predictions must be nonnegative.")
    return base if cutoff is None else np.minimum(base, cutoff)


def _probability(value, n):
    probability = _vector(value, "selection_probability")
    if len(probability) != n or np.any((probability <= 0) | (probability > 1)):
        raise ValueError("selection_probability must be aligned and in (0,1]; no positivity clipping.")
    return probability


@dataclass(frozen=True)
class SurvivalConformalCalibration:
    """Lower event-time bounds, not calibrated survival curves or upper bounds."""

    calibration: ConformalCalibration
    cutoff: float | None
    selected_subject_id: np.ndarray
    n_original: int

    def __post_init__(self):
        ids = np.array(self.selected_subject_id, copy=True)
        ids.setflags(write=False)
        object.__setattr__(self, "selected_subject_id", ids)

    def __reduce__(self):
        return type(self), (self.calibration, self.cutoff, self.selected_subject_id, self.n_original)

    def interval(self, prediction, *, selection_probability=None):
        level = self.calibration.level
        base = _base(prediction, level, self.cutoff)
        if self.cutoff is None:
            if selection_probability is not None:
                raise ValueError("Selection probabilities require cutoff calibration.")
            ratio = None
        else:
            if selection_probability is None:
                raise ValueError("Cutoff prediction requires explicit selection_probability.")
            with np.errstate(over="raise", divide="raise"):
                ratio = 1/_probability(selection_probability, len(base))
        correction = self.calibration.corrections(ratio)
        with np.errstate(over="raise", invalid="raise"):
            lower = np.maximum(0, base-correction)
        if self.cutoff is not None:
            lower = np.minimum(lower, self.cutoff)
        coverage = ("conservative_marginal_for_event_time" if self.cutoff is None else
                    "marginal_with_known_censoring_selection" if self.calibration.weight_source == "known" else
                    "approximate_with_estimated_censoring_selection")
        return ConformalInterval(lower, np.full(len(base), np.inf), level,
            "survival_observed_lower" if self.cutoff is None else "survival_cutoff_lower",
            len(self.calibration.scores), correction, coverage=coverage,
            uncertainty_kind="future_survival_time_conformal", scope="one_event_time_lower_bound",
            weight_source=None if self.cutoff is None else self.calibration.weight_source)


def calibrate_survival(target, prediction, *, level=.9, cutoff=None, censoring_time=None,
                       selection_probability=None, weight_source="known"):
    """Calibrate right-censored data without treating censoring as exact events.

    Default: score all observed min(T,C) values; the lower bound also covers T
    conservatively under exchangeability, without independent-censoring claims.
    Cutoff: retain known C>=cutoff and calibrate exact min(T,cutoff), weighted
    by 1/P(C>=cutoff|X). Requires conditional independence T and C given X and
    externally supplied selection probabilities, independent of calibration.
    Known probabilities give finite-sample marginal coverage; estimates do not
    automatically give exact or doubly-robust guarantees. Cutoff must be chosen
    independently of calibration outcomes. No delayed entry, adaptive cutoffs,
    ordinary random-censoring imputation, upper bounds or curve calibration.
    """
    level = _level(level)
    if not isinstance(target, SurvivalTarget):
        raise TypeError("Survival calibration requires SurvivalTarget.")
    if np.any(target.entry != 0):
        raise NotImplementedError("Delayed-entry survival conformal is unsupported.")
    if cutoff is None:
        if censoring_time is not None or selection_probability is not None or weight_source != "known":
            raise ValueError("Censoring selection arguments require an explicit cutoff.")
        selected, ratio = np.ones(len(target), dtype=bool), None
    else:
        if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, Real):
            raise ValueError("cutoff must be a finite positive scalar.")
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be a finite positive scalar.")
        if censoring_time is None or selection_probability is None:
            raise ValueError("Cutoff calibration needs known censoring_time for every subject and selection_probability.")
        censoring = np.asarray(censoring_time, dtype=float)
        if (censoring.shape != target.time.shape or np.isnan(censoring).any() or np.any(censoring <= 0)
                or np.any(target.time > censoring)
                or np.any(target.time[~target.event] != censoring[~target.event])):
            raise ValueError("Known censoring times must agree with observations and be supplied even for events.")
        selected = censoring >= cutoff
        if not selected.any():
            raise ValueError("No calibration subjects have C>=cutoff.")
        probability = _probability(selection_probability, len(target))
        with np.errstate(over="raise", divide="raise"):
            ratio = 1/probability[selected]
    base = _base(prediction, level, cutoff)
    if len(base) != len(target):
        raise ValueError("Survival predictions and targets must align.")
    response = target.time if cutoff is None else np.minimum(target.time, cutoff)
    state = ConformalCalibration((base-response)[selected], "quantile", level, ratio, weight_source)
    return SurvivalConformalCalibration(state, cutoff, target.subject_id[selected], len(target))


@dataclass(frozen=True)
class SurvivalConformalPredictor:
    """Backend-neutral fitted snapshot plus survival lower-bound calibration."""

    _predictor: Any
    calibration: SurvivalConformalCalibration

    def predict_interval(self, X, *, selection_probability=None, **predict_kwargs):
        prediction = self._predictor.predict_distribution(X, **predict_kwargs)
        if not isinstance(prediction, SurvivalPrediction) or prediction.n_subjects != len(X):
            raise NotImplementedError("A row-aligned shared SurvivalPrediction is required.")
        return self.calibration.interval(prediction, selection_probability=selection_probability)

    def _artifact_state(self):
        return dict(self.__dict__)

    def save_model(self, path):
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        return load_estimator(path, cls)


def calibrate_survival_estimator(estimator, X, target, *, predict_kwargs=None, **calibration_kwargs):
    """Snapshot a fitted survival model; no fitting, splitting or censoring estimation."""
    if getattr(estimator, "_estimator_type", None) != "survival":
        raise NotImplementedError("Use a fitted survival estimator, not an uncensored regression model.")
    if len(X) != len(target):
        raise ValueError("Survival features and targets must align.")
    predictor = snapshot_estimator(estimator)
    prediction = predictor.predict_distribution(X, **dict(predict_kwargs or {}))
    if not isinstance(prediction, SurvivalPrediction):
        raise NotImplementedError("The estimator must provide shared SurvivalPrediction.")
    return SurvivalConformalPredictor(predictor, calibrate_survival(target, prediction, **calibration_kwargs))
