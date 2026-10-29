"""Descriptive curves and censor-aware metrics, independent of fitting engines.

IPCW uses an explicitly supplied training censoring estimate. It never fits
censoring on evaluation outcomes. Marginal censoring independence is assumed;
conditional censoring models, IPCW with entry, and Uno concordance are unsupported.

References
----------
- Kaplan, E. L., and Meier, P. (1958). Nonparametric Estimation from Incomplete
  Observations.
  https://doi.org/10.2307/2281868
- Aalen, O. (1978). Nonparametric Inference for a Family of Counting Processes.
  https://doi.org/10.1214/aos/1176344247
- Gerds, T. A., and Schumacher, M. (2006). Consistent estimation of the expected Brier
  score in general survival models with right-censored event times.
  https://doi.org/10.1002/bimj.200610301
- Harrell, F. E., Jr., et al. (1982). Evaluating the Yield of Medical Tests.
  https://doi.org/10.1001/jama.1982.03320430047030

The implemented IPCW estimator assumes marginal censoring independence; the cited
general conditional-censoring extensions do not expand that support.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .contracts import SurvivalTarget, _readonly, validate_weights
from .prediction import SurvivalPrediction


def _subject_data(target, sample_weight):
    from .longitudinal import LongitudinalSurvivalTarget

    if isinstance(target, LongitudinalSurvivalTarget):
        weights = validate_weights(target, sample_weight, allow_longitudinal=True)
        first = np.r_[
            0, np.flatnonzero(target.subject_id[1:] != target.subject_id[:-1]) + 1
        ]
        return target.subject_outcomes(), weights[first]
    return target, validate_weights(target, sample_weight)


def _no_entry(target):
    if not isinstance(target, SurvivalTarget):
        raise NotImplementedError(
            "Evaluation requires a SurvivalTarget or a complete terminal-event longitudinal target."
        )
    if np.any(target.entry > 0):
        raise NotImplementedError(
            "Delayed entry is unsupported for IPCW and concordance."
        )


def _step(time, values, query, *, initial=1.0):
    query = np.asarray(query, dtype=float)
    if not np.isfinite(query).all() or np.any(query < 0):
        raise ValueError("Evaluation times must be finite and nonnegative.")
    if np.any(query > time[-1]):
        raise ValueError("Evaluation extends beyond observed follow-up support.")
    return np.r_[initial, values][np.searchsorted(time, query, side="right")]


@dataclass(frozen=True)
class SurvivalCurve:
    """Weighted KM/NA descriptive curve, with entry < t <= exit risk sets.

    Arrays are snapshots at distinct positive-weight exit times. Event and
    censor ties share the risk set; events precede censoring. Survival is the
    product-limit KM estimate, not exp(-Nelson-Aalen). No confidence bands or
    population survival before the entry mechanism are inferred.
    """

    time: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray
    survival: np.ndarray
    cumulative_hazard: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=float))

    def survival_at(self, times):
        """Right-continuous KM values; no extrapolation past final follow-up."""
        return _step(self.time, self.survival, times)

    def cumulative_hazard_at(self, times):
        """Right-continuous Nelson-Aalen values on observed support."""
        return _step(self.time, self.cumulative_hazard, times, initial=0.0)


def survival_curve(target, *, sample_weight=None):
    """Entry-aware Kaplan-Meier and Nelson-Aalen, without changing fitted models.

    Weights are case/frequency weights. Zero-weight records do not define
    follow-up support. No survey or clustered variance estimate is provided.
    """
    target, weights = _subject_data(target, sample_weight)
    active = weights > 0
    exit_time, entry = target.time[active], target.entry[active]
    weights, event = weights[active], target.event[active]
    time, groups = np.unique(exit_time, return_inverse=True)
    events = np.bincount(groups, weights=weights * event, minlength=len(time))
    censored = np.bincount(groups, weights=weights * ~event, minlength=len(time))
    order = np.argsort(entry)
    entry_weight = np.r_[0, np.cumsum(weights[order])]
    entered = entry_weight[np.searchsorted(entry[order], time, side="left")]
    exited = np.r_[0, np.cumsum(events + censored)[:-1]]
    risk = entered - exited
    if np.any(risk <= 0):
        raise FloatingPointError("Nonpositive descriptive risk set.")
    increment = events / risk
    return SurvivalCurve(
        time, risk, events, censored, np.cumprod(1 - increment), np.cumsum(increment)
    )


@dataclass(frozen=True)
class CensoringDistribution:
    """Training reverse-KM snapshot; positive values on finite follow-up only.

    At a tied exit, observed events are removed before the censoring jump.
    Evaluation uses right-continuous G(t), including at observed event times.
    Training and evaluation targets must have entry zero. No clipping of G,
    tail extrapolation, automatic held-out fitting, or covariate adjustment.
    """

    time: np.ndarray
    probability: np.ndarray

    def __post_init__(self):
        time, probability = np.asarray(self.time), np.asarray(self.probability)
        if (
            time.ndim != 1
            or not len(time)
            or probability.shape != time.shape
            or not np.isfinite(time).all()
            or np.any(time <= 0)
            or np.any(np.diff(time) <= 0)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or np.any(np.diff(probability) > 0)
        ):
            raise ValueError("Invalid reverse-KM step function.")
        object.__setattr__(self, "time", _readonly(time, dtype=float))
        object.__setattr__(self, "probability", _readonly(probability, dtype=float))

    @classmethod
    def fit(cls, training_target, *, sample_weight=None):
        """Estimate marginal censoring from training outcomes supplied explicitly."""
        training_target, sample_weight = _subject_data(training_target, sample_weight)
        _no_entry(training_target)
        curve = survival_curve(training_target, sample_weight=sample_weight)
        risk = curve.at_risk - curve.events
        increment = np.divide(
            curve.censored, risk, out=np.zeros_like(risk), where=risk > 0
        )
        return cls(curve.time, np.cumprod(1 - increment))

    def survival_at(self, times):
        """G(t); errors on zero censoring survival or unsupported follow-up."""
        probability = _step(self.time, self.probability, times)
        if np.any(probability <= 0):
            raise ValueError("IPCW requires strictly positive censoring survival.")
        return probability


@dataclass(frozen=True)
class ConcordanceResult:
    """Harrell concordance and pair-weight totals (unit weights give counts)."""

    concordance: float
    concordant: float
    discordant: float
    tied_risk: float
    comparable: float
    tied_time: float


def concordance_index(target, risk, *, higher_risk, sample_weight=None, tied_tol=1e-8):
    """Harrell C; require an explicit risk direction, never infer one from a law.

    An event compares with later exits and censored exits at the same time.
    Event/event ties are excluded. Risk ties within tied_tol contribute half.
    Pair weights are products of case weights; no comparable pairs is an error.
    This is not IPCW concordance and may depend on the censoring distribution.
    """
    target, weights = _subject_data(target, sample_weight)
    _no_entry(target)
    risk = np.asarray(risk, dtype=float)
    if risk.shape != target.time.shape or not np.isfinite(risk).all():
        raise ValueError("risk must be a finite vector aligned with target.")
    if not isinstance(higher_risk, (bool, np.bool_)):
        raise ValueError("higher_risk must explicitly be True or False.")
    if not np.isfinite(tied_tol) or tied_tol < 0:
        raise ValueError("tied_tol must be finite and nonnegative.")
    concordant = discordant = tied = tied_time = 0.0
    for i in np.flatnonzero(target.event & (weights > 0)):
        equal_time = (target.time == target.time[i]) & ~target.event
        comparable = (target.time > target.time[i]) | equal_time
        pair = weights[i] * weights[comparable]
        difference = risk[i] - risk[comparable]
        if not higher_risk:
            difference = -difference
        tied += pair[np.abs(difference) <= tied_tol].sum()
        concordant += pair[difference > tied_tol].sum()
        discordant += pair[difference < -tied_tol].sum()
        tied_time += weights[i] * weights[equal_time].sum()
    count = concordant + discordant + tied
    if count == 0:
        raise ValueError("No comparable positive-weight pairs.")
    return ConcordanceResult(
        (concordant + 0.5 * tied) / count,
        concordant,
        discordant,
        tied,
        count,
        tied_time,
    )


@dataclass(frozen=True)
class BrierScore:
    """IPCW errors at explicit times, and finite-grid normalized integration."""

    time: np.ndarray
    score: np.ndarray

    def __post_init__(self):
        for name in ("time", "score"):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=float))

    def integrated(self):
        """Trapezoidal average over [time[0],time[-1]], not from zero implicitly."""
        if len(self.time) < 2:
            raise ValueError("Integrated Brier score requires at least two times.")
        return float(trapezoid(self.score, self.time) / (self.time[-1] - self.time[0]))


def brier_score(prediction, target, times, *, censoring, sample_weight=None):
    """IPCW Brier curve from a prediction and explicit training reverse-KM.

    Cases at/before t contribute S(t)^2/G(exit); subjects observed beyond t
    contribute (1-S(t))^2/G(t). Earlier censoring contributes zero, without
    removing its case weight from the denominator. No delayed entry support.
    """
    target, weights = _subject_data(target, sample_weight)
    _no_entry(target)
    if not isinstance(prediction, SurvivalPrediction) or prediction.n_subjects != len(
        target
    ):
        raise ValueError("SurvivalPrediction rows must align with target.")
    if not isinstance(censoring, CensoringDistribution):
        raise TypeError("Supply a training CensoringDistribution explicitly.")
    times = prediction._times(times)
    if times[-1] >= np.max(target.time[weights > 0]):
        raise ValueError(
            "Brier times must precede final positive-weight evaluation follow-up."
        )
    g_time = censoring.survival_at(times)
    event = target.event & (weights > 0) & (target.time <= times[-1])
    inverse = np.zeros(len(target))
    inverse[event] = 1 / censoring.survival_at(target.time[event])
    survival = prediction.survival(times)
    if not np.isfinite(survival).all() or np.any((survival < 0) | (survival > 1)):
        raise FloatingPointError("Brier predictions must be finite probabilities.")
    cases = (target.time[:, None] <= times) & target.event[:, None]
    controls = target.time[:, None] > times
    errors = (
        cases * survival**2 * inverse[:, None] + controls * (1 - survival) ** 2 / g_time
    )
    return BrierScore(times, weights @ errors / weights.sum())
