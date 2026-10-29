"""Complete subject trajectories and conditional piecewise prediction.

Rows describe (start,stop] intervals with covariates known/predictable over that
interval. No last-observation-carried-forward, gap filling, forecasting or
resampling of partial subject histories is performed.
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .contracts import SurvivalTarget, _readonly


@dataclass(frozen=True)
class SurvivalTrajectory:
    """Ordered contiguous covariate records, kept together by subject_id.

    Delayed entry is allowed for fitting. Prediction of unconditional survival
    requires supplied covariate histories from zero through a common horizon.
    Subject blocks may be in any order but cannot be interleaved.
    """

    start: np.ndarray
    stop: np.ndarray
    subject_id: np.ndarray

    def __post_init__(self):
        start, stop = (
            _readonly(self.start, dtype=float),
            _readonly(self.stop, dtype=float),
        )
        ids = _readonly(self.subject_id, dtype=object)
        if (
            start.ndim != 1
            or not len(start)
            or stop.shape != start.shape
            or ids.shape != start.shape
            or not np.isfinite(start).all()
            or not np.isfinite(stop).all()
            or np.any(start < 0)
            or np.any(stop <= start)
            or pd.isna(ids).any()
        ):
            raise ValueError(
                "Trajectory rows require aligned finite 0<=start<stop and nonmissing subject IDs."
            )
        codes, _ = pd.factorize(ids, sort=False)
        boundaries = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        if len(np.unique(codes[boundaries])) != len(boundaries):
            raise ValueError(
                "Each trajectory subject must form one contiguous ordered block."
            )
        same = codes[1:] == codes[:-1]
        if np.any(start[1:][same] != stop[:-1][same]):
            raise ValueError(
                "Within-subject histories must be ordered and contiguous: gaps/overlaps are unsupported."
            )
        for name, value in (("start", start), ("stop", stop), ("subject_id", ids)):
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.start)

    @property
    def subject_ids(self):
        return _readonly(pd.unique(self.subject_id))

    @property
    def n_subjects(self):
        return len(self.subject_ids)

    @property
    def normalization(self):
        codes, _ = pd.factorize(self.subject_id, sort=False)
        return _readonly(1 / np.bincount(codes)[codes])

    def record_indices(self, subject_ids):
        """Indices of complete selected subjects, in original record order."""
        requested = np.asarray(subject_ids, dtype=object)
        if (
            requested.ndim != 1
            or not len(requested)
            or pd.Index(requested).has_duplicates
            or not np.isin(requested, self.subject_ids).all()
        ):
            raise ValueError("Select nonempty known subject IDs, not record indices.")
        return _readonly(np.flatnonzero(np.isin(self.subject_id, requested)))

    def split_subjects(self, *, validation_fraction=0.2, seed=None):
        """Return disjoint complete training/validation targets in source order.

        Select aligned feature/weight/offset rows with record_indices on the
        original target. No record-level randomization or outcome stratification.
        """
        if (
            not np.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1
            or self.n_subjects < 2
        ):
            raise ValueError(
                "Subject splitting needs at least two subjects and a fraction in (0,1)."
            )
        n_validation = int(np.ceil(self.n_subjects * validation_fraction))
        if n_validation >= self.n_subjects:
            raise ValueError("Subject split would leave an empty training cohort.")
        selected = np.random.default_rng(seed).permutation(self.n_subjects)
        return (
            self.subset(self.subject_ids[selected[n_validation:]]),
            self.subset(self.subject_ids[selected[:n_validation]]),
        )

    def resample_subjects(self, indices):
        """Return (target, source_record_indices) for sampled subject occurrences.

        Repetitions preserve every record of each selected subject and receive
        fresh occurrence IDs. The caller retains the source-subject selection
        as provenance; case weights and offsets index by the returned rows.
        """
        from uuid import uuid4

        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or np.any(indices < 0)
            or np.any(indices >= self.n_subjects)
        ):
            raise ValueError(
                "Resample nonempty integer subject indices within the cohort."
            )
        blocks = [
            np.flatnonzero(self.subject_id == subject) for subject in self.subject_ids
        ]
        rows = np.concatenate([blocks[i] for i in indices])
        namespace = uuid4().hex
        ids = np.concatenate(
            [
                np.repeat(f"{namespace}:{occurrence}", len(blocks[i]))
                for occurrence, i in enumerate(indices)
            ]
        )
        values = {field.name: getattr(self, field.name)[rows] for field in fields(self)}
        values["subject_id"] = ids
        return type(self)(**values), _readonly(rows)

    def subset(self, subject_ids):
        """Select complete subjects, retaining original record and subject order."""
        take = self.record_indices(subject_ids)
        return type(self)(
            **{field.name: getattr(self, field.name)[take] for field in fields(self)}
        )


@dataclass(frozen=True)
class LongitudinalSurvivalTarget(SurvivalTrajectory):
    """One terminal event/right-censoring per complete trajectory.

    event is 0 on every nonfinal record; the final record is an event or censor.
    Recurrent events, intermittent risk gaps and simultaneous records are not
    represented by this contract. subject_id is required and can repeat.
    """

    event: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        event = np.asarray(self.event)
        if event.shape != self.start.shape or not np.isin(event, [0, 1]).all():
            raise ValueError("Trajectory event must be an aligned 0/1 vector.")
        nonfinal = np.r_[self.subject_id[:-1] == self.subject_id[1:], False]
        if np.any(event[nonfinal]):
            raise NotImplementedError(
                "Only a terminal event on the final subject record is supported."
            )
        object.__setattr__(self, "event", _readonly(event, dtype=bool))

    @property
    def has_explicit_ids(self):
        return True

    def subject_outcomes(self):
        """One terminal outcome per subject, preserving first delayed entry.

        This collapses outcome bookkeeping only. Prediction still requires
        supplied covariate paths; it never treats records as static covariates.
        """
        first = np.r_[
            0, np.flatnonzero(self.subject_id[1:] != self.subject_id[:-1]) + 1
        ]
        last = np.r_[first[1:] - 1, len(self) - 1]
        return SurvivalTarget(
            self.stop[last],
            self.event[last],
            self.start[first],
            subject_id=self.subject_ids,
        )

    def as_segments(self):
        """Explicit conditional risk-record targets for existing likelihood kernels.

        Segment IDs identify records, not independent patients. Group ownership
        and subject-normalized scoring remain with the longitudinal contract.
        """
        return SurvivalTarget(self.stop, self.event, entry=self.start)


def validate_record_weights(target, weights):
    """Case weights travel with subjects and must agree across their records."""
    codes, unique = pd.factorize(target.subject_id, sort=False)
    for i in range(len(unique)):
        values = weights[codes == i]
        if np.any(values != values[0]):
            raise ValueError(
                "Longitudinal case weights must be constant within each subject's records."
            )


def subject_weight_sum(target, weights):
    return (
        float(weights @ target.normalization)
        if isinstance(target, LongitudinalSurvivalTarget)
        else float(weights.sum())
    )


def predict_trajectory(estimator, X, trajectory, *, offset=None, **predict_kwargs):
    """Shared piecewise survival conditional on explicitly supplied future paths.

    Output rows follow trajectory.subject_ids, with the union of model cuts and
    covariate-change times. Every subject must supply a complete path from zero
    to the same finite horizon within fitted support. No extrapolation, partial
    paths, trajectory uncertainty, component intervals or parametric time-varying
    parameter paths are supplied. Ordinary predict_distribution remains a
    hypothetical constant-covariate path, one distribution per supplied row.
    """
    from .prediction import SurvivalPrediction

    if not isinstance(trajectory, SurvivalTrajectory) or len(X) != len(trajectory):
        raise ValueError("X must align with a SurvivalTrajectory.")
    if not callable(getattr(estimator, "survival_time_grid", None)) or not hasattr(
        estimator, "predictor_backend"
    ):
        raise NotImplementedError(
            "Estimator has no shared piecewise trajectory prediction adapter."
        )
    grid = estimator.survival_time_grid()
    eta = np.asarray(estimator.predict_link(X, offset=offset, **predict_kwargs))
    if eta.shape != (len(trajectory), len(grid) - 1) or not np.isfinite(eta).all():
        raise ValueError(
            "Piecewise predictors must align with trajectory records and fitted intervals."
        )
    cuts, records, interval = _trajectory_layout(grid, trajectory)
    return SurvivalPrediction(
        np.exp(eta[records, interval]), cuts, backend=estimator.predictor_backend
    )


def _trajectory_layout(grid, trajectory, *, cuts=None):
    """Record/model-interval lookup on an explicit complete path grid."""
    ids = trajectory.subject_ids
    stops = [trajectory.stop[trajectory.subject_id == subject][-1] for subject in ids]
    starts = [trajectory.start[trajectory.subject_id == subject][0] for subject in ids]
    if np.any(np.asarray(starts) != 0) or np.any(np.asarray(stops) != stops[0]):
        raise NotImplementedError(
            "Prediction needs complete paths from zero to one common horizon."
        )
    horizon = stops[0]
    if horizon > grid[-1]:
        raise NotImplementedError(
            "Trajectory horizon extends beyond fitted piecewise support."
        )
    if cuts is not None and (cuts[0] != 0 or cuts[-1] != horizon):
        raise ValueError(
            "Query/reference trajectories must share one prediction horizon."
        )
    cuts = np.unique(
        np.r_[
            0.0,
            grid[(grid > 0) & (grid < horizon)],
            trajectory.stop,
            [] if cuts is None else cuts,
        ]
    )
    interval = np.searchsorted(grid, cuts[1:], side="left") - 1
    records = np.empty((len(ids), len(cuts) - 1), dtype=int)
    for i, subject in enumerate(ids):
        rows = np.flatnonzero(trajectory.subject_id == subject)
        records[i] = rows[np.searchsorted(trajectory.stop[rows], cuts[1:], side="left")]
    return cuts, records, interval
