"""Observation-aware residual calculations, independent of either fitter."""

from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy

from .contracts import SurvivalTarget, _readonly
from .prediction import SurvivalPrediction


@dataclass(frozen=True)
class SurvivalResiduals:
    """Unweighted rowwise residuals; event flags retain Cox-Snell censoring.

    Cox-Snell values H(exit)-H(entry) are right censored when event is false;
    they must not be used as fully observed exponential or normal samples.
    Martingale/deviance are counting-process diagnostics, not Pearson errors,
    inferential standard errors, or Cox Schoenfeld residuals.
    """

    cox_snell: np.ndarray
    martingale: np.ndarray
    deviance: np.ndarray
    event: np.ndarray
    subject_id: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def survival_residuals(prediction, target):
    """Cox-Snell, martingale and deviance residuals, conditional on entry.

    With r=H(exit)-H(entry), M=event-r and
    D=sign(M)*sqrt(-2*(M+event*log(r))). No weighting or probability clipping;
    a zero-hazard event gives infinite deviance. Censored zero exposure gives 0.
    Randomized PIT, uncensored worm plots and Cox-specific diagnostics are
    deliberately not substituted for these observation-aware calculations.
    """
    from .longitudinal import LongitudinalSurvivalTarget

    longitudinal = isinstance(target, LongitudinalSurvivalTarget)
    original = target
    if longitudinal:
        target = target.as_segments()
    if not isinstance(target, SurvivalTarget):
        raise TypeError("target must be SurvivalTarget.")
    if not isinstance(prediction, SurvivalPrediction) or prediction.n_subjects != len(
        target
    ):
        raise ValueError("SurvivalPrediction rows must align with target.")
    residual = prediction.cumulative_hazard_at(target.time, conditional_on=target.entry)
    if longitudinal:
        first = np.r_[
            0, np.flatnonzero(original.subject_id[1:] != original.subject_id[:-1]) + 1
        ]
        residual = np.add.reduceat(residual, first)
        target = original.subject_outcomes()
    if not np.isfinite(residual).all() or np.any(residual < 0):
        raise FloatingPointError(
            "Cox-Snell residuals require finite nonnegative hazard increments."
        )
    martingale = target.event - residual
    # xlogy defines the censored r=0 term as zero rather than 0*log(0).
    squared = -2 * (martingale + xlogy(target.event, residual))
    # r-1-log(r) can lose roundoff near one; only its nonnegative square root
    # is rounded at zero. Observations and hazard increments are never clipped.
    deviance = np.sign(martingale) * np.sqrt(np.maximum(squared, 0))
    return SurvivalResiduals(
        residual, martingale, deviance, target.event, target.subject_id
    )
