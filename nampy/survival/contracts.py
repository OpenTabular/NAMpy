"""Backend-neutral single-event survival targets and interval designs."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..contracts import AdditivePrediction

if TYPE_CHECKING:
    from .longitudinal import LongitudinalSurvivalTarget


def _readonly(value, *, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, init=False)
class SurvivalTarget:
    """One terminal event or right censoring per subject, with optional entry.

    ``event`` is explicitly 0/1. Entry conditions the likelihood on survival
    to that time; it is never an observation weight or a model feature.
    """

    time: np.ndarray
    event: np.ndarray
    entry: np.ndarray
    subject_id: np.ndarray
    has_explicit_ids: bool = field(init=False, repr=False)

    def __init__(self, time, event, entry=None, subject_id=None):
        for name, value in (
            ("time", time),
            ("event", event),
            ("entry", entry),
            ("subject_id", subject_id),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self):
        object.__setattr__(self, "has_explicit_ids", self.subject_id is not None)
        time = _readonly(self.time, dtype=float)
        event = np.asarray(self.event)
        if time.ndim != 1 or not len(time):
            raise ValueError("time must be a nonempty vector.")
        if event.shape != time.shape or not np.all(np.isin(event, [0, 1])):
            raise ValueError("event must be an aligned vector containing only 0/1.")
        entry = _readonly(
            np.zeros(len(time)) if self.entry is None else self.entry, dtype=float
        )
        if (
            entry.shape != time.shape
            or not np.isfinite(time).all()
            or not np.isfinite(entry).all()
            or np.any(entry < 0)
            or np.any(time <= entry)
        ):
            raise ValueError("Finite times must satisfy 0 <= entry < time.")
        ids = _readonly(
            np.arange(len(time)) if self.subject_id is None else self.subject_id
        )
        if (
            ids.shape != time.shape
            or pd.isna(ids).any()
            or pd.Index(ids).has_duplicates
        ):
            raise ValueError("subject_id must be unique, nonmissing and aligned.")
        for name, value in (
            ("time", time),
            ("event", _readonly(event, dtype=bool)),
            ("entry", entry),
            ("subject_id", ids),
        ):
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.time)

    def take(self, indices):
        indices = np.atleast_1d(indices)
        return type(self)(
            self.time[indices],
            self.event[indices],
            self.entry[indices],
            self.subject_id[indices],
        )

    @classmethod
    def from_frame(cls, frame, *, time, event, entry=None, subject_id=None):
        """Construct from explicit outcome column names; never infer coding."""
        return cls(
            frame[time],
            frame[event],
            None if entry is None else frame[entry],
            None if subject_id is None else frame[subject_id],
        )


def validate_weights(target, sample_weight=None, *, require_event=False, allow_longitudinal=False):
    """Validate subject likelihood weights, independently of event status."""
    from .longitudinal import LongitudinalSurvivalTarget, validate_record_weights
    longitudinal = isinstance(target, LongitudinalSurvivalTarget)
    if not isinstance(target, SurvivalTarget) and not (allow_longitudinal and longitudinal):
        raise TypeError(
            "y must be SurvivalTarget; other censoring modes are unsupported."
        )
    weights = (
        np.ones(len(target))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        weights.shape != (len(target),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "sample_weight must be finite, nonnegative and aligned with positive sum."
        )
    if require_event and not np.any(target.event & (weights > 0)):
        raise ValueError("Fitting requires at least one positive-weight event.")
    if longitudinal:
        validate_record_weights(target, weights)
    return weights.copy()


@dataclass(frozen=True)
class PiecewiseExponentialDesign:
    """Subject-by-interval exposure and event counts, without duplicated X."""

    target: "SurvivalTarget | LongitudinalSurvivalTarget"
    cuts: np.ndarray
    exposure: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)

    def __post_init__(self):
        from .longitudinal import LongitudinalSurvivalTarget
        target = self.target.as_segments() if isinstance(self.target, LongitudinalSurvivalTarget) else self.target
        if not isinstance(target, SurvivalTarget):
            raise TypeError("target must be SurvivalTarget or LongitudinalSurvivalTarget.")
        cuts = validate_cuts(self.cuts)
        if target.time.max() > cuts[-1]:
            raise ValueError("cuts must cover every observed exit time.")
        exposure = np.clip(
            np.minimum(target.time[:, None], cuts[1:])
            - np.maximum(target.entry[:, None], cuts[:-1]),
            0,
            None,
        )
        events = np.zeros_like(exposure)
        interval = np.asarray(np.searchsorted(cuts, target.time, side="left")) - 1
        rows = np.flatnonzero(target.event)
        events[rows, interval[rows]] = 1
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(self, "exposure", _readonly(exposure))
        object.__setattr__(self, "events", _readonly(events))

    def validate_fit(self, sample_weight=None, *, saturated_baseline=True):
        weights = validate_weights(self.target, sample_weight, require_event=True, allow_longitudinal=True)
        if np.any(weights @ self.exposure <= 0):
            raise ValueError("Every interval needs positive-weight exposure.")
        if saturated_baseline and np.any(weights @ self.events <= 0):
            raise NotImplementedError(
                "Zero-event interval baseline boundary fits are unsupported; "
                "choose cuts with positive-weight events in each interval."
            )
        return weights

    def log_likelihood(self, log_hazard):
        """Continuous event-time contributions, conditional on entry."""
        eta = np.asarray(log_hazard, dtype=float)
        if eta.shape != self.exposure.shape or not np.isfinite(eta).all():
            raise ValueError(
                "log_hazard must be finite with shape (n_subjects, n_intervals)."
            )
        with np.errstate(over="raise", invalid="raise"):
            # Only evaluate cumulative hazard where the subject is at risk.
            active = self.exposure > 0
            integrated = np.zeros_like(eta)
            integrated[active] = self.exposure[active] * np.exp(eta[active])
        return np.sum(self.events * eta - integrated, axis=1)


def validate_cuts(cuts):
    cuts = _readonly(cuts, dtype=float)
    if (
        cuts.ndim != 1
        or len(cuts) < 2
        or not np.isfinite(cuts).all()
        or cuts[0] != 0
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError(
            "Estimator cuts must be finite, strictly increasing and start at zero."
        )
    return cuts


@dataclass(frozen=True)
class SurvivalComponents:
    """Predictor decomposition on subject-by-interval rows and its mapping."""

    prediction: AdditivePrediction
    subject_index: np.ndarray
    interval: np.ndarray
    evaluation_time: np.ndarray

    def __post_init__(self):
        n = len(self.prediction.link)
        for name, dtype in (
            ("subject_index", int),
            ("interval", int),
            ("evaluation_time", float),
        ):
            value = _readonly(getattr(self, name), dtype=dtype)
            if value.shape != (n,):
                raise ValueError(
                    "Survival component row mappings must align with predictors."
                )
            object.__setattr__(self, name, value)
