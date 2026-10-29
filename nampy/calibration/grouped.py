"""Separate fixed-stratum and whole-group exchangeability calibration targets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conformal import (
    ConformalCalibration,
    ConformalInterval,
    _bounds,
    calibrate_conformal,
)


def _groups(values, n):
    labels = np.array(values, dtype=object, copy=True)
    if labels.shape != (n,) or pd.isna(labels).any():
        raise ValueError("groups must be aligned nonmissing scalar labels.")
    try:
        codes, unique = pd.factorize(labels, sort=False)
    except TypeError as error:
        raise ValueError("Group labels must be hashable scalars.") from error
    labels.setflags(write=False)
    return labels, codes, tuple(unique)


@dataclass(frozen=True)
class GroupedConformalCalibration:
    """Fixed-stratum quantiles or maxima over complete exchangeable groups.

    Strata must be an externally fixed covariate partition, not outcomes.
    Cluster mode treats each complete group as one exchangeable observation;
    its interval covers all members of one new group simultaneously. It does
    not cover all future groups jointly, or a new member of a seen group.
    """

    labels: tuple
    states: tuple
    mode: str

    @property
    def level(self):
        return self.states[0].level

    @property
    def method(self):
        return self.states[0].method

    def interval(self, prediction, *, groups):
        lower, upper = _bounds(prediction, self.method, self.level)
        _, codes, unique = _groups(groups, len(lower))
        if self.mode == "cluster":
            if set(unique) & set(self.labels):
                raise NotImplementedError("Cluster prediction requires entirely new group IDs.")
            correction = self.states[0].correction
            counts = np.full(len(lower), len(self.labels))
            coverage, scope = "marginal_over_exchangeable_groups", "simultaneous_within_one_new_group"
        else:
            if not set(unique) <= set(self.labels):
                raise NotImplementedError("Unseen strata have no calibration; pooled fallback is unsupported.")
            lookup = dict(zip(self.labels, self.states, strict=True))
            correction = np.array([lookup[unique[c]].correction for c in codes])
            counts = np.array([len(lookup[unique[c]].scores) for c in codes])
            coverage, scope = "within_fixed_stratum_under_exchangeability", "one_observation_within_stratum"
        with np.errstate(over="raise", invalid="raise"):
            lo, hi = lower-correction, upper+correction
        return ConformalInterval(lo, hi, self.level, self.method,
            sum(len(state.scores) for state in self.states), correction,
            coverage=coverage, scope=scope, calibration_counts=counts)


def calibrate_grouped(y, prediction, groups, *, mode="cluster", method="absolute", level=.9):
    """Calibrate complete groups or fixed strata; these are distinct guarantees.

    No density-ratio/group combinations, outcome-defined/adaptive strata,
    partial groups, hierarchical random-member methods or time-series methods.
    Training/validation/calibration group disjointness is caller-owned.
    """
    if mode not in {"cluster", "stratum"}:
        raise ValueError("Grouped calibration mode must be cluster or stratum.")
    state = calibrate_conformal(y, prediction, method=method, level=level)
    _, codes, unique = _groups(groups, len(state.scores))
    states: tuple[ConformalCalibration, ...]
    if mode == "cluster":
        scores = [state.scores[codes == i].max() for i in range(len(unique))]
        states = (ConformalCalibration(np.asarray(scores), method, level),)
    else:
        states = tuple(ConformalCalibration(state.scores[codes == i], method, level) for i in range(len(unique)))
    return GroupedConformalCalibration(unique, states, mode)
