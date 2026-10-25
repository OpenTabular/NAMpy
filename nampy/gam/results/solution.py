"""Stable fitted-output contract shared across fitting backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitResult:
    """Stable numerical fit outputs shared across statistical solvers.

    Stores coefficients, fitted predictors, effective degrees of freedom, and
    covariance matrices. :class:`nampy.gam.results.GAMFitResult` wraps this
    record as ``core`` and adds criterion and per-term reporting information.
    :class:`nampy.gam.results.GAMResult` aggregates the compiled design, core
    solution, and reporting summary for a fitted model.
    """

    coef_full: np.ndarray
    intercept: float
    beta: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    rss: float | None
    deviance: float
    edf: float
    trace_H: float
    scale: float
    cov_bayes: np.ndarray | None
    cov_freq: np.ndarray | None
    cov_unconditional: np.ndarray | None
    H_coef: np.ndarray
    edf2: np.ndarray | None = None
    edf_by_term: np.ndarray | None = None
    penalty_quadratic: float | None = None
    loglik: float | None = None
    converged: bool | None = None
    iter: int | None = None
    failed_step: bool | None = None
    failure_reason: str | None = None
    inner_trace: list | None = None
    coef_space: str = "fit"
    cov_bayes_space: str = "fit"
    cov_freq_space: str = "fit"
    cov_unconditional_space: str = "fit"
    coef_optimization: np.ndarray | None = None
    cov_bayes_optimization: np.ndarray | None = None
    cov_freq_optimization: np.ndarray | None = None
    positive_coefficient_mask: np.ndarray | None = None


__all__ = ["FitResult"]
