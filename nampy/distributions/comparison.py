"""Explicit candidate fits and shared held-out normalized probability scoring."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..diagnostics.distribution import (
    CentileCalibration,
    _centile_probabilities,
    centile_calibration,
)
from .contracts import DistributionFitStatus
from .prediction import DistributionPrediction


@dataclass(frozen=True)
class DistributionCandidate:
    """Estimator template and its backend-specific fit/prediction arguments.

    Every template is sklearn-cloned before fitting. Common training weights
    belong to compare_distributions, not fit_kwargs. Context/offset arrays stay
    explicit here. status_reader is an optional backend adapter for external
    estimators; it must return DistributionFitStatus. Without an adapter or a
    distribution_fit_status method, a fit stays unknown and cannot be ranked.
    """

    estimator: Any
    fit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    predict_kwargs: Mapping[str, Any] = field(default_factory=dict)
    status_reader: Callable[[Any], DistributionFitStatus] | None = None

    def __post_init__(self):
        fit_kwargs, predict_kwargs = dict(self.fit_kwargs), dict(self.predict_kwargs)
        if {"X", "y", "data", "sample_weight"} & fit_kwargs.keys():
            raise ValueError(
                "Candidate fit_kwargs cannot replace common data or sample_weight."
            )
        if {"X", "data"} & predict_kwargs.keys():
            raise ValueError(
                "Candidate predict_kwargs cannot replace common validation rows."
            )
        if self.status_reader is not None and not callable(self.status_reader):
            raise TypeError("status_reader must be callable.")
        object.__setattr__(
            self, "fit_kwargs", MappingProxyType(copy.deepcopy(fit_kwargs))
        )
        object.__setattr__(
            self, "predict_kwargs", MappingProxyType(copy.deepcopy(predict_kwargs))
        )


@dataclass(frozen=True)
class DistributionComparisonRow:
    """One candidate, including failed/ineligible fits and optional diagnostics."""

    name: str
    fit_status: DistributionFitStatus
    status: str
    eligible: bool
    rank: int | None
    mean_log_prob: float | None
    n_validation: int
    n_scored: int
    n_zero_probability: int
    validation_weight_sum: float
    effective_n: float
    fit_seconds: float
    evaluation_seconds: float
    warnings: tuple[str, ...] = ()
    failure_stage: str | None = None
    failure_reason: str | None = None
    measure: tuple[str, tuple[float, ...]] | None = None
    calibration: CentileCalibration | None = None
    calibration_error: str | None = None
    log_prob: np.ndarray | None = field(default=None, repr=False, compare=False)
    estimator: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class DistributionComparison:
    """Original candidate order, ranking and independently fitted estimator clones.

    mean_log_prob uses common positive-weight observations and normalized
    validation weights. -inf is a valid zero-probability score and ranks last;
    NaN/+inf scores are ineligible. Exact ties share their minimum rank.
    Calibration is unweighted on positive-weight rows, separately labelled in
    to_frame; missing operations or diagnostic failures do not invalidate a
    normalized likelihood score. This is a selection result, not an unbiased
    estimate of the selected model's generalization performance.
    """

    results: tuple[DistributionComparisonRow, ...]
    response_units: str | None = None

    @property
    def ranking(self) -> tuple[str, ...]:
        rows = [row for row in self.results if row.eligible]
        return tuple(
            row.name for row in sorted(rows, key=lambda row: cast(int, row.rank))
        )

    @property
    def models(self) -> dict[str, Any]:
        """Clones, including partially fitted models retained for inspection."""
        return {
            row.name: row.estimator for row in self.results if row.estimator is not None
        }

    def to_frame(self):
        """Return a flat ranking table; calibration details stay in results."""
        import pandas as pd

        records = []
        for row in self.results:
            records.append(
                {
                    "name": row.name,
                    "response_units": self.response_units,
                    "rank": row.rank,
                    "eligible": row.eligible,
                    "status": row.status,
                    "backend": row.fit_status.backend,
                    "fit_state": row.fit_status.state,
                    "fit_message": row.fit_status.message,
                    "mean_log_prob": row.mean_log_prob,
                    "n_validation": row.n_validation,
                    "n_scored": row.n_scored,
                    "n_zero_probability": row.n_zero_probability,
                    "validation_weight_sum": row.validation_weight_sum,
                    "effective_n": row.effective_n,
                    "fit_seconds": row.fit_seconds,
                    "evaluation_seconds": row.evaluation_seconds,
                    "failure_stage": row.failure_stage,
                    "failure_reason": row.failure_reason,
                    "warnings": row.warnings,
                    "measure": row.measure,
                    "calibration_weighting": "unweighted_positive_weight_rows"
                    if row.calibration is not None
                    else None,
                    "calibration_error": row.calibration_error,
                }
            )
        frame = pd.DataFrame.from_records(records)
        frame["rank"] = pd.array(frame["rank"], dtype="Int64")
        return frame.sort_values("rank", kind="stable", na_position="last").reset_index(
            drop=True
        )


def _response(y, name):
    value = np.array(y, dtype=float, copy=True)
    if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a nonempty finite scalar-response vector.")
    return value


def _weights(values, n, name):
    weights = np.ones(n) if values is None else np.array(values, dtype=float, copy=True)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"{name} must be a finite nonnegative vector of shape ({n},).")
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError(f"{name} must have a positive finite sum.")
    return weights


def _measure(distribution):
    spec = distribution.spec
    if spec.response_kind == "mixed":
        atoms = getattr(spec, "atom_locations", None)
        if atoms is None:
            raise NotImplementedError(
                "Mixed-law comparison requires declared atom_locations for a common reference measure."
            )
        return ("mixed", atoms)
    return (spec.response_kind, ())


def compare_distributions(
    candidates: Mapping[str, Any],
    X_train,
    y_train,
    X_valid,
    y_valid,
    *,
    sample_weight=None,
    validation_weight=None,
    calibration_probs=(0.05, 0.5, 0.95),
    response_units: str | None = None,
) -> DistributionComparison:
    """Fit explicit candidates and compare normalized held-out log probabilities.

    Supply named estimator templates or DistributionCandidate configurations.
    Training and validation sets are explicit; no split, family search, refit,
    response transform, Jacobian correction or estimator.score dispatch is done.
    All candidates receive isolated copies of identical inputs. Fit kwargs can
    configure inner validation, but the comparison holdout is never passed to
    fitting automatically. Required context/offsets belong to each candidate.

    Every positive-validation-weight row must be scored in its original order;
    there is no per-model row deletion or common-intersection fallback. Rows of
    weight zero do not enter the score or unweighted calibration. Continuous,
    discrete and mixed measures cannot be ranked together; mixed laws also need
    identical declared atoms. Incompatible eligible groups are all unranked.
    Failed, nonconverged, unknown-status and invalid-evaluation candidates remain
    in results. Unknown external backends need an explicit status adapter.

    Use an untouched test set or nested selection after choosing a family.
    Response units and normalization are the probability law's contract; custom
    laws must retain any response-transform Jacobian. No cross-backend EDF/AIC
    or convergence equivalence is claimed. Execution is sequential.
    """
    from sklearn.base import clone

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a nonempty named mapping.")
    if any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("Candidate names must be nonempty strings.")
    if response_units is not None and (
        not isinstance(response_units, str) or not response_units
    ):
        raise ValueError("response_units must be a nonempty label or None.")
    train_y, valid_y = _response(y_train, "y_train"), _response(y_valid, "y_valid")
    if (
        X_train is None
        or len(X_train) != len(train_y)
        or X_valid is None
        or len(X_valid) != len(valid_y)
    ):
        raise ValueError(
            "Explicit feature and response rows must have matching lengths for each split."
        )
    training_weights = _weights(sample_weight, len(train_y), "sample_weight")
    weights = _weights(validation_weight, len(valid_y), "validation_weight")
    active = weights > 0
    normalized = weights[active] / weights.sum()
    n_active, weight_sum = int(active.sum()), float(weights.sum())
    effective_n = float(1 / np.sum(normalized**2))
    probabilities = (
        None if calibration_probs is None else _centile_probabilities(calibration_probs)
    )
    configs = {
        name: candidate
        if isinstance(candidate, DistributionCandidate)
        else DistributionCandidate(candidate)
        for name, candidate in candidates.items()
    }
    rows = []
    for name, config in configs.items():
        model = None
        fit_status = DistributionFitStatus(
            "unknown", "unknown", "Fit has not completed."
        )
        status, reason, failure_stage = "fit_error", None, None
        score, log_prob, measure, calibration, calibration_error = (
            None,
            None,
            None,
            None,
            None,
        )
        n_scored = n_zero = 0
        fit_seconds = evaluation_seconds = 0.0
        stage = "clone"
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = clone(config.estimator)
                if model is config.estimator:
                    model = None
                    raise TypeError(
                        "Candidate cloning reused the input template; independent fits are required."
                    )
                stage = "fit"
                kwargs = copy.deepcopy(dict(config.fit_kwargs))
                if sample_weight is not None:
                    kwargs["sample_weight"] = training_weights.copy()
                model.fit(copy.deepcopy(X_train), train_y.copy(), **kwargs)
                stage = "status"
                reader = config.status_reader
                if reader is not None:
                    reported_status = reader(model)
                elif callable(getattr(model, "distribution_fit_status", None)):
                    reported_status = model.distribution_fit_status()
                else:
                    reported_status = DistributionFitStatus(
                        "external",
                        "unknown",
                        "No distribution_fit_status adapter; supply status_reader explicitly.",
                    )
                if not isinstance(reported_status, DistributionFitStatus):
                    raise TypeError(
                        "Fit-status adapters must return DistributionFitStatus."
                    )
                fit_status = reported_status
                fit_seconds = perf_counter() - started
                if not fit_status.eligible:
                    status, reason, failure_stage = (
                        "ineligible_fit",
                        fit_status.message,
                        "status",
                    )
                else:
                    started = perf_counter()
                    stage = "prediction"
                    distribution = model.predict_distribution(
                        copy.deepcopy(X_valid),
                        **copy.deepcopy(dict(config.predict_kwargs)),
                    )
                    if not isinstance(distribution, DistributionPrediction):
                        raise TypeError(
                            "predict_distribution must return DistributionPrediction."
                        )
                    if distribution.n_rows != len(valid_y):
                        raise ValueError(
                            "Prediction rows do not match the full validation split; omission/reordering is unsupported."
                        )
                    measure = _measure(distribution)
                    stage = "score"
                    log_prob = np.array(
                        distribution.log_prob(valid_y), dtype=float, copy=True
                    )
                    selected = log_prob[active]
                    n_scored = int(np.sum(~np.isnan(selected) & ~np.isposinf(selected)))
                    n_zero = int(np.isneginf(selected).sum())
                    if n_scored != n_active:
                        raise ValueError(
                            "Every positive-weight validation row requires a defined log probability; NaN/+inf are ineligible."
                        )
                    score = -np.inf if n_zero else float(np.sum(normalized * selected))
                    status = "ok"
                    if probabilities is not None:
                        if {"cdf", "quantile"} <= distribution.spec.operations:
                            try:
                                calibration = centile_calibration(
                                    distribution,
                                    np.where(active, valid_y, np.nan),
                                    probs=probabilities,
                                )
                            except Exception as error:
                                calibration_error = f"{type(error).__name__}: {error}"
                        else:
                            calibration_error = "Calibration requires CDF and quantile operations; score remains available."
                    evaluation_seconds = perf_counter() - started
            except Exception as error:
                failure_stage, reason = stage, f"{type(error).__name__}: {error}"
                if stage in {"clone", "fit", "status"}:
                    fit_seconds = perf_counter() - started
                    fit_status = DistributionFitStatus(
                        fit_status.backend, "failed", reason
                    )
                    status = "fit_error"
                else:
                    evaluation_seconds = perf_counter() - started
                    status = "evaluation_error"
            messages = tuple(
                f"{type(item.message).__name__}: {item.message}" for item in caught
            )
        if log_prob is not None:
            log_prob.setflags(write=False)
        rows.append(
            DistributionComparisonRow(
                name=name,
                fit_status=fit_status,
                status=status,
                eligible=status == "ok",
                rank=None,
                mean_log_prob=score,
                n_validation=n_active,
                n_scored=n_scored,
                n_zero_probability=n_zero,
                validation_weight_sum=weight_sum,
                effective_n=effective_n,
                fit_seconds=fit_seconds,
                evaluation_seconds=evaluation_seconds,
                warnings=messages,
                failure_stage=failure_stage,
                failure_reason=reason,
                measure=measure,
                calibration=calibration,
                calibration_error=calibration_error,
                log_prob=log_prob,
                estimator=model,
            )
        )
    measures = {row.measure for row in rows if row.eligible}
    if len(measures) > 1:
        rows = [
            replace(
                row,
                eligible=False,
                status="incompatible_measure",
                failure_stage="comparison",
                failure_reason="Candidates use incompatible reference measures; compare kinds/atom sets separately.",
            )
            if row.eligible
            else row
            for row in rows
        ]
    ranked = sorted(
        (i for i, row in enumerate(rows) if row.eligible),
        key=lambda i: -cast(float, rows[i].mean_log_prob),
    )
    previous, rank = None, 0
    for position, index in enumerate(ranked, 1):
        if position == 1 or rows[index].mean_log_prob != previous:
            rank = position
        previous = rows[index].mean_log_prob
        rows[index] = replace(rows[index], rank=rank)
    return DistributionComparison(tuple(rows), response_units)
