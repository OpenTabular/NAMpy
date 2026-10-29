"""Cox-specific residual data in mgcv's coefficient coordinates and tie order."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoxDiagnostics:
    time: np.ndarray
    event: np.ndarray
    subject_id: np.ndarray
    schoenfeld: np.ndarray
    cumulative_score: np.ndarray
    martingale: np.ndarray
    deviance: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def cox_diagnostics(matrix, beta, target, strata, offset, baseline):
    """Unscaled source Schoenfeld and drift-corrected cumulative score process.

    mgcv hazard() computes prefix-weighted means in descending record order,
    including its tie-order convention. These are not substituted for the
    complete-tied-risk-set coefficient score used during fitting. No PH test
    p-values, smooth-term covariance scaling or influence approximation.
    """
    with np.errstate(over="raise", invalid="raise"):
        gamma = np.exp(matrix @ beta + offset)
    schoenfeld = np.full_like(matrix, np.nan)
    score = np.zeros_like(matrix)
    hazard = np.zeros(len(target))
    for label in np.unique(strata):
        rows = np.flatnonzero(strata == label)
        rows = rows[np.argsort(-target.time[rows], kind="stable")]
        mean = (
            np.cumsum(gamma[rows, None] * matrix[rows], axis=0)
            / np.cumsum(gamma[rows])[:, None]
        )
        raw = (matrix[rows] - mean) * target.event[rows, None]
        cumulative = np.cumsum(raw[::-1], axis=0)[::-1]
        events = np.flatnonzero(target.event[rows])
        if len(events) < 2:
            raise NotImplementedError(
                "Source Cox score-process drift correction requires two events per diagnostic stratum."
            )
        drift = np.linspace(1.0, 0.0, len(events))[
            np.clip(
                np.searchsorted(events, np.arange(len(rows)), side="right") - 1,
                0,
                len(events) - 1,
            )
        ]
        score[rows] = cumulative - drift[:, None] * cumulative[0]
        raw[~target.event[rows]] = np.nan
        schoenfeld[rows] = raw
        times, values = baseline[int(label)]
        hazard[rows] = (
            values[np.searchsorted(times, target.time[rows], side="right") - 1]
            * gamma[rows]
        )
    martingale = target.event - hazard
    # Source residuals() protects log survival at -1e-50 for deviance only.
    log_survival = np.minimum(-hazard, -1e-50)
    deviance = np.sign(martingale) * np.sqrt(
        -2 * (martingale + target.event * np.log(-log_survival))
    )
    return CoxDiagnostics(
        target.time,
        target.event,
        target.subject_id,
        schoenfeld,
        score,
        martingale,
        deviance,
    )
