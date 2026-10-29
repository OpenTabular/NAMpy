"""Ordinary subject-pairs resampling, fresh backend refits and auditable outcomes.

No trainer or fitted estimator is retained in a bootstrap result. The source
estimator is never refitted. Neural routes own training configuration; shared
code owns row resampling, failure policy and interval calculations.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .contracts import SurvivalTarget, _readonly, validate_weights
from .interpretation import SurvivalEffects, prepare_survival_effects
from .prediction import SurvivalPrediction
from .uncertainty import (
    SurvivalCurveInterval,
    SurvivalEffectInterval,
    _curve_interval,
    _functional_interval,
    _functional_values,
    _simultaneous_limits,
)


@dataclass(frozen=True)
class SurvivalBootstrapFailure:
    replicate: int
    stage: str
    error_type: str
    message: str


class SurvivalBootstrapError(RuntimeError):
    """A failed attempt; the original exception is retained as its cause."""

    def __init__(self, failure):
        self.failure = failure
        super().__init__(
            f"Bootstrap replicate {failure.replicate} failed during "
            f"{failure.stage}: {failure.error_type}: {failure.message}"
        )


def _check_effects(current, expected):
    if (
        tuple(current.effects) != tuple(expected.effects)
        or dict(current.features) != dict(expected.features)
        or current.predictor_names != expected.predictor_names
        or not np.array_equal(current.time, expected.time)
        or not np.array_equal(current.conditional_on, expected.conditional_on)
    ):
        raise NotImplementedError(
            "Bootstrap effects require the same semantic term layout, subjects, grid and conditioning."
        )
    valid = current.time[None, :] >= current.conditional_on[:, None]
    values = (
        current.reference_survival,
        *current.survival.values(),
        *current.effects.values(),
    )
    if any(
        value.shape != expected.reference_survival.shape
        or not np.isfinite(value[valid]).all()
        for value in values
    ):
        raise FloatingPointError(
            "Bootstrap effects require finite aligned values on the declared valid grid."
        )


@dataclass(frozen=True)
class SurvivalBootstrapResult:
    """Prediction snapshots plus subject-resampling and training provenance.

    Indices refer to independent subjects (subject_ids for longitudinal targets),
    rather than longitudinal record rows; duplicate occurrences receive fresh
    local IDs during fitting, without discarding their source identity here.
    Failed attempts are never replaced. Intervals refuse any failed replicate.
    """

    point_prediction: SurvivalPrediction
    predictions: tuple
    resample_indices: np.ndarray
    source_subject_id: np.ndarray
    successful_replicates: tuple
    failures: tuple
    training_seeds: tuple
    training_policy: str
    validation_policy: str
    point_effects: SurvivalEffects | None = None
    effect_samples: tuple = ()

    def __post_init__(self):
        object.__setattr__(
            self, "resample_indices", _readonly(self.resample_indices, dtype=int)
        )
        object.__setattr__(self, "source_subject_id", _readonly(self.source_subject_id))
        for name in (
            "predictions",
            "successful_replicates",
            "failures",
            "training_seeds",
            "effect_samples",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def uncertainty_kind(self):
        return (
            "subject_bootstrap_with_training_randomness"
            if self.training_policy == "independent_seed"
            else "subject_bootstrap"
        )

    def _require_complete(self):
        if self.failures:
            raise ValueError(
                "Bootstrap intervals require every requested replicate to succeed; inspect failures."
            )
        if len(self.predictions) < 2:
            raise ValueError(
                "Bootstrap intervals require at least two successful refits."
            )

    def percentile_interval(
        self, *, grid, estimand="survival", conditional_on=None, level=0.95
    ):
        """Pointwise R type-7 percentile limits from every requested subject refit.

        Grid means times for survival/hazard and restriction horizons for RMST.
        The estimate is the original fitted functional, not the bootstrap mean.
        Integration errors propagate; failed refits cannot be omitted.
        Simultaneous/effect operations are separate from these percentile limits.
        """
        self._require_complete()
        if estimand not in {"survival", "hazard", "rmst"}:
            raise NotImplementedError(
                "Bootstrap percentile intervals support survival, hazard or RMST only."
            )
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie in (0,1).")
        point = self.point_prediction
        grid = point._times(grid)
        landmark = point._landmarks(conditional_on, np.inf)
        if np.any(landmark > point.support[1]):
            raise NotImplementedError("Landmark lies beyond supported cuts.")

        def evaluate(prediction):
            return (
                prediction.survival(grid, conditional_on=landmark)
                if estimand == "survival"
                else _functional_values(prediction, grid, landmark, estimand)
            )

        estimate = evaluate(point)
        values = np.stack([evaluate(value) for value in self.predictions])
        valid = grid[None, :] >= landmark[:, None]
        if (
            not np.isfinite(estimate[valid]).all()
            or not np.isfinite(values[:, valid]).all()
        ):
            raise FloatingPointError(
                "Bootstrap intervals require finite valid functional predictions."
            )
        lower, upper = np.full(estimate.shape, np.nan), np.full(estimate.shape, np.nan)
        lower[valid], upper[valid] = np.quantile(
            values[:, valid], [(1 - level) / 2, (1 + level) / 2], axis=0
        )
        if estimand == "rmst":
            width = np.maximum(grid[None, :] - landmark[:, None], 0)
            lower, upper = (
                np.minimum(np.maximum(lower, 0), width),
                np.minimum(np.maximum(upper, 0), width),
            )
        return SurvivalCurveInterval(
            grid,
            estimate,
            lower,
            upper,
            landmark,
            level,
            len(self.predictions),
            self.uncertainty_kind,
            "pointwise_percentile",
            transformation=estimand,
            estimand=(
                "restricted_mean_remaining_lifetime" if estimand == "rmst" else estimand
            ),
        )

    def simultaneous_interval(
        self, *, grid, estimand="survival", conditional_on=None, level=0.95
    ):
        """Centered maximum-standardized bootstrap band on the supplied finite grid.

        Coverage is joint over all valid subjects/times (or RMST horizons).
        Empirical bootstrap SDs provide a common cellwise scale, not nested
        studentization. Critical values use R type 8. Single-fit coefficient,
        replicated-fit and subject-bootstrap uncertainty remain distinct.
        """
        self._require_complete()
        if estimand not in {"survival", "hazard", "rmst"}:
            raise NotImplementedError(
                "Bootstrap simultaneous intervals support survival, hazard or RMST only."
            )
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie in (0,1).")
        prediction = self.point_prediction
        grid = prediction._times(grid)
        landmark = prediction._landmarks(conditional_on, np.inf)
        if np.any(landmark > prediction.support[1]):
            raise NotImplementedError("Landmark lies beyond supported cuts.")
        if estimand == "survival":
            result = _curve_interval(
                self.predictions,
                grid,
                point_prediction=prediction,
                level=level,
                conditional_on=landmark,
                uncertainty_kind=self.uncertainty_kind,
                coverage="simultaneous_bootstrap",
                interval="simultaneous",
            )
        else:
            result = _functional_interval(
                prediction,
                self.predictions,
                grid,
                landmark,
                estimand,
                level,
                "simultaneous",
                self.uncertainty_kind,
            )
        return replace(result, coverage="simultaneous_bootstrap")

    def effect_interval(self, *, level=0.95, interval="pointwise"):
        """Paired reference-plus-term contrasts captured inside each subject refit.

        The reference cohort, weights, grid and selected semantic term layout
        are fixed by bootstrap_survival(effects=...). Every refit recenters its
        own contributions and evaluates both contrast sides together. Joint
        coverage includes all selected terms/subjects/valid times, not one band
        per term. Reference-population resampling is not implemented.
        """
        self._require_complete()
        if self.point_effects is None or len(self.effect_samples) != len(
            self.predictions
        ):
            raise ValueError(
                "Capture effects during bootstrap refitting with effects={...}."
            )
        if (
            interval not in {"pointwise", "simultaneous"}
            or not np.isfinite(level)
            or not 0 < level < 1
        ):
            raise ValueError(
                "Select pointwise/simultaneous effect intervals and level in (0,1)."
            )
        prepared = self.point_effects
        names = tuple(prepared.effects)
        point = np.stack([prepared.effects[name] for name in names])
        values = np.stack(
            [
                np.stack([sample.effects[name] for name in names])
                for sample in self.effect_samples
            ]
        )
        valid = np.broadcast_to(
            prepared.time[None, None, :] >= prepared.conditional_on[None, :, None],
            point.shape,
        )
        if (
            not np.isfinite(point[valid]).all()
            or not np.isfinite(values[:, valid]).all()
        ):
            raise FloatingPointError(
                "Bootstrap effects require finite valid paired contrasts."
            )
        se, critical = None, None
        if interval == "simultaneous":
            lower, upper, se, critical = _simultaneous_limits(
                point, values, valid, level
            )
            lower, upper = np.clip(lower, -1, 1), np.clip(upper, -1, 1)
        else:
            lower, upper = np.full(point.shape, np.nan), np.full(point.shape, np.nan)
            lower[valid], upper[valid] = np.quantile(
                values[:, valid], [(1 - level) / 2, (1 + level) / 2], axis=0
            )
        return SurvivalEffectInterval(
            prepared,
            {
                name: SurvivalCurveInterval(
                    prepared.time,
                    point[i],
                    lower[i],
                    upper[i],
                    prepared.conditional_on,
                    level,
                    len(self.effect_samples),
                    self.uncertainty_kind,
                    "simultaneous_bootstrap"
                    if interval == "simultaneous"
                    else "pointwise_percentile",
                    None if se is None else se[i],
                    critical,
                    "survival_difference",
                    "all_selected_terms_subjects_and_times"
                    if interval == "simultaneous"
                    else "pointwise",
                    "reference_plus_term_survival_difference",
                )
                for i, name in enumerate(names)
            },
        )


def _rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices].reset_index(drop=True).copy(deep=True)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("Features must be a two-dimensional array or DataFrame.")
    return values[indices].copy()


def _aligned_offset(value, n, name):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if result.ndim not in {1, 2} or len(result) != n or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and aligned with subject rows.")
    return result.copy()


def bootstrap_survival(
    estimator,
    X,
    y,
    X_predict,
    *,
    n_resamples=200,
    seed=None,
    resample_indices=None,
    sample_weight=None,
    offset=None,
    prediction_offset=None,
    X_val=None,
    y_val=None,
    sample_weight_val=None,
    offset_val=None,
    training_randomness="fixed",
    on_failure="raise",
    effects=None,
    prediction_trajectory=None,
    fit_context=None,
    prediction_context=None,
):
    """Ordinary subject-pairs bootstrap of an already fitted survival estimator.

    Supply original training data and fixed prediction rows explicitly. Uniform
    sampling selects independent subjects, carrying every longitudinal record,
    event/entry, case weight and offset
    together; weights are not sampling probabilities or virtual subjects.
    Explicit integer indices (B,n) override n_resamples and enable external
    reference checks. Each sample is physically reconstructed and recompiled.

    Neural refits require a fixed disjoint validation cohort with explicit IDs
    on both original targets. Duplicate sampled occurrences never cross an
    automatic train/validation split. Fixed training seeds isolate resampling
    conditional on the configured algorithm; independent seeds label combined
    sampling/training variation, without a variance decomposition. GPU/numerical
    nondeterminism is not eliminated by a fixed seed.

    Failures raise by default; 'record' attempts the preselected samples without
    retry and returns failure provenance, but refuses all interval inference if
    any attempt failed. effects optionally supplies prepare_survival_effects
    keyword arguments (except offset, taken from prediction_offset); times must
    be supplied before refitting because fitted replicas are not retained.
    Reference rows/weights are fixed across refits. Missing/incompatible terms
    fail that replicate, without zero-padding or a sparse union. No
    stratification, parametric bootstrap, warm starts, BCa/basic/studentized or
    continuous-domain intervals.
    """
    from .longitudinal import (
        LongitudinalSurvivalTarget,
        SurvivalTrajectory,
        predict_trajectory,
    )

    longitudinal = isinstance(y, LongitudinalSurvivalTarget)
    weights = validate_weights(y, sample_weight, allow_longitudinal=longitudinal)
    n_subjects = y.n_subjects if longitudinal else len(y)
    source_ids = y.subject_ids if longitudinal else y.subject_id
    if prediction_trajectory is not None:
        if not isinstance(prediction_trajectory, SurvivalTrajectory) or len(
            prediction_trajectory
        ) != len(X_predict):
            raise ValueError(
                "prediction_trajectory must align with prediction feature records."
            )
    allowed_context = set(getattr(estimator, "survival_bootstrap_context_names", ()))
    contexts = []
    for context, size in ((fit_context, len(y)), (prediction_context, len(X_predict))):
        context = {} if context is None else dict(context)
        if set(context) - allowed_context:
            raise NotImplementedError(
                "Unsupported bootstrap observation context for this estimator."
            )
        if any(np.shape(value) != (size,) for value in context.values()):
            raise ValueError(
                "Bootstrap context arrays must align with their feature rows."
            )
        contexts.append(
            {name: np.array(value, copy=True) for name, value in context.items()}
        )
    fit_context, prediction_context = contexts
    if (
        hasattr(estimator, "strata_")
        and len(estimator.strata_) > 1
        and ("strata" not in fit_context or "strata" not in prediction_context)
    ):
        raise ValueError(
            "Stratified Cox bootstrap needs explicit training and prediction strata context."
        )

    def predict(model):
        if prediction_trajectory is not None:
            return predict_trajectory(
                model,
                X_predict,
                prediction_trajectory,
                offset=prediction_offset,
                **prediction_context,
            )
        return model.predict_distribution(
            X_predict, offset=prediction_offset, **prediction_context
        )

    if len(X) != len(y):
        raise ValueError("Training features and target must align.")
    if training_randomness not in {"fixed", "independent"} or on_failure not in {
        "raise",
        "record",
    }:
        raise ValueError(
            "Select fixed/independent training randomness and raise/record failure handling."
        )
    if not callable(getattr(estimator, "clone_for_survival_bootstrap", None)):
        raise NotImplementedError(
            "Estimator has no supported subject-bootstrap refit adapter."
        )
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be supplied together.")
    validation = y_val is not None
    if validation:
        if isinstance(y_val, LongitudinalSurvivalTarget) != longitudinal:
            raise ValueError(
                "Training and validation target record layouts must agree."
            )
        validate_weights(y_val, sample_weight_val, allow_longitudinal=longitudinal)
        if len(X_val) != len(y_val):
            raise ValueError("Validation features and target must align.")
        if not y.has_explicit_ids or not y_val.has_explicit_ids:
            raise ValueError(
                "Bootstrap validation requires explicit source subject IDs on both targets."
            )
        if np.isin(y.subject_id, y_val.subject_id).any():
            raise ValueError("Bootstrap training and validation source IDs overlap.")
    elif offset_val is not None or sample_weight_val is not None:
        raise ValueError("Validation offsets/weights require validation data.")
    offset = _aligned_offset(offset, len(y), "offset")
    offset_val = (
        _aligned_offset(offset_val, len(y_val), "offset_val") if validation else None
    )
    point = predict(estimator)
    expected_rows = (
        prediction_trajectory.n_subjects
        if prediction_trajectory is not None
        else len(X_predict)
    )
    if not isinstance(point, SurvivalPrediction) or point.n_subjects != expected_rows:
        raise ValueError(
            "Point survival predictions must align with prediction subjects."
        )
    status = estimator.distribution_fit_status()
    if not status.eligible:
        raise ValueError(
            f"Original fitted survival estimator is ineligible: {status.message}"
        )
    point_effects = None
    effect_options = None
    if effects is not None:
        effect_options = dict(effects)
        if prediction_trajectory is not None:
            if "trajectory" in effect_options:
                raise ValueError(
                    "Bootstrap query trajectory comes from prediction_trajectory."
                )
            effect_options["trajectory"] = prediction_trajectory
        if "offset" in effect_options:
            raise ValueError(
                "Bootstrap effect offsets come from prediction_offset; use reference_offset for reference data."
            )
        point_effects = prepare_survival_effects(
            estimator, X_predict, offset=prediction_offset, **effect_options
        )
        _check_effects(point_effects, point_effects)
        # Keep the declared point term set rather than silently selecting a
        # different set when a resampled design loses feature/term support.
        effect_options.update(
            times=point_effects.time, terms=tuple(point_effects.effects)
        )
    plan_seed, fit_seed = np.random.SeedSequence(seed).spawn(2)
    if resample_indices is None:
        if (
            isinstance(n_resamples, (bool, np.bool_))
            or not isinstance(n_resamples, (int, np.integer))
            or n_resamples < 2
        ):
            raise ValueError("n_resamples must be an integer >= 2.")
        indices = np.random.default_rng(plan_seed).integers(
            n_subjects, size=(n_resamples, n_subjects)
        )
    else:
        indices = np.asarray(resample_indices)
        if (
            indices.ndim != 2
            or indices.shape[1] != n_subjects
            or len(indices) < 2
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or np.any(indices < 0)
            or np.any(indices >= n_subjects)
        ):
            raise ValueError(
                "resample_indices must be integer (B,n), B>=2, within original subject rows."
            )
        indices = indices.copy()
    seeds = np.random.default_rng(fit_seed).integers(0, 2**31 - 1, size=len(indices))
    predictions, successes, failures, effective_seeds, effect_samples = (
        [],
        [],
        [],
        [],
        [],
    )
    policy = None
    with TemporaryDirectory(prefix="nampy-survival-bootstrap-") as directory:
        for b, selected in enumerate(indices):
            replica, current_policy, effective_seed = (
                estimator.clone_for_survival_bootstrap(
                    y,
                    training_randomness=training_randomness,
                    seed=int(seeds[b]),
                    checkpoint_path=Path(directory) / f"replicate-{b}",
                    has_validation=validation,
                )
            )
            if (
                replica is estimator
                or current_policy
                not in {"not_applicable", "fixed_seed", "independent_seed"}
                or (policy is not None and policy != current_policy)
            ):
                raise RuntimeError(
                    "Bootstrap adapter must return fresh estimators with a consistent training policy."
                )
            policy = current_policy
            effective_seeds.append(effective_seed)
            # IDs here identify distinct sampled occurrences; the source map is
            # retained in the result and validated before any neural split.
            if longitudinal:
                target, selected_rows = y.resample_subjects(selected)
            else:
                selected_rows = selected
                target = SurvivalTarget(
                    y.time[selected], y.event[selected], y.entry[selected]
                )
            features = _rows(X, selected_rows)
            known = None if offset is None else offset[selected_rows].copy()
            weight = None if sample_weight is None else weights[selected_rows].copy()
            validation_kwargs = (
                {
                    "X_val": _rows(X_val, np.arange(len(y_val))),
                    "y_val": y_val,
                    "offset_val": offset_val,
                    "sample_weight_val": sample_weight_val,
                }
                if validation
                else {}
            )
            stage = "fit"
            try:
                fit_options = {
                    name: value[selected_rows] for name, value in fit_context.items()
                }
                if weight is not None:
                    fit_options["sample_weight"] = weight
                replica.fit(
                    features, target, offset=known, **fit_options, **validation_kwargs
                )
                stage = "fit_status"
                fitted_status = replica.distribution_fit_status()
                if not fitted_status.eligible:
                    raise RuntimeError(
                        f"Ineligible bootstrap fit: {fitted_status.message}"
                    )
                stage = "training_likelihood"
                score_options = {
                    name: value[selected_rows] for name, value in fit_context.items()
                }
                if weight is not None:
                    score_options["sample_weight"] = weight
                if not np.isfinite(
                    replica.score(features, target, offset=known, **score_options)
                ):
                    raise FloatingPointError(
                        "Bootstrap observed-event training likelihood is nonfinite."
                    )
                stage = "prediction"
                prediction = predict(replica)
                if (
                    not isinstance(prediction, SurvivalPrediction)
                    or prediction.n_subjects != point.n_subjects
                ):
                    raise ValueError(
                        "Bootstrap predictions must preserve prediction subject rows."
                    )
                prepared = None
                if effect_options is not None:
                    assert point_effects is not None
                    stage = "effect_preparation"
                    prepared = prepare_survival_effects(
                        replica,
                        _rows(X_predict, np.arange(len(X_predict))),
                        offset=prediction_offset,
                        **effect_options,
                    )
                    _check_effects(prepared, point_effects)
                predictions.append(prediction)
                successes.append(b)
                if prepared is not None:
                    effect_samples.append(prepared)
            except Exception as error:
                failure = SurvivalBootstrapFailure(
                    b, stage, type(error).__name__, str(error)
                )
                if on_failure == "raise":
                    raise SurvivalBootstrapError(failure) from error
                failures.append(failure)
    assert policy is not None
    return SurvivalBootstrapResult(
        point,
        tuple(predictions),
        indices,
        source_ids,
        tuple(successes),
        tuple(failures),
        tuple(effective_seeds),
        policy,
        "fixed_external" if validation else "not_used",
        point_effects,
        tuple(effect_samples),
    )
