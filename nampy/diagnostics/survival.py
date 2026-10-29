"""Shared survival effect preparation; no backend or rendering dependencies."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import xlogy

from .._data import frame_for_estimator
from ..contracts import AdditivePrediction
from ..diagnostics.contributions import center_additive_prediction
from ..survival.contracts import SurvivalComponents, SurvivalTarget, _readonly
from ..survival.prediction import SurvivalPrediction


def _subject_components(components, n_subjects):
    if isinstance(components, AdditivePrediction):
        if len(components.link) != n_subjects:
            raise ValueError("Predictor rows must align with subjects.")
        return components
    if not isinstance(components, SurvivalComponents):
        raise TypeError("Survival effects require additive predictor components.")
    prediction = components.prediction
    if prediction.link.shape[1] != 1:
        raise NotImplementedError(
            "Expanded survival components require one log-hazard predictor."
        )
    k = int(np.max(components.interval)) + 1
    rows, intervals = components.subject_index, components.interval
    indices = rows * k + intervals
    if (
        np.any(rows < 0)
        or np.any(rows >= n_subjects)
        or np.any(intervals < 0)
        or len(indices) != n_subjects * k
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "Expanded components must cover each subject/interval exactly once."
        )

    def unpack(values):
        result = np.empty((n_subjects, k), dtype=np.asarray(values).dtype)
        result[rows, intervals] = np.broadcast_to(values, prediction.link.shape)[:, 0]
        return result

    return AdditivePrediction(
        link=unpack(prediction.link),
        terms={name: unpack(value) for name, value in prediction.terms.items()},
        intercept=unpack(prediction.intercept),
        offset=None if prediction.offset is None else unpack(prediction.offset),
        backend=prediction.backend,
        term_metadata=prediction.term_metadata,
        predictor_names=tuple(f"log_hazard[{j}]" for j in range(k)),
    )


@dataclass(frozen=True)
class SurvivalEffects:
    """Prepared reference-plus-term survival and contrasts, each (n,t).

    Contrasts are nonlinear survival differences, not an additive decomposition
    of survival. Feature arrays retain all coordinates of interactions. Known
    offsets stay at their supplied subject values; reference weights center
    terms, not event times. No coefficient or response-spread bands are implied.
    """

    time: np.ndarray
    reference_survival: np.ndarray
    survival: dict
    effects: dict
    feature_values: dict
    features: dict
    predictor_names: tuple
    conditional_on: np.ndarray
    backend: str

    def __post_init__(self):
        for name in ("time", "reference_survival", "conditional_on"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        for name in ("survival", "effects", "feature_values"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        key: _readonly(value)
                        for key, value in getattr(self, name).items()
                    }
                ),
            )
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


def prepare_survival_effects(
    estimator,
    X,
    *,
    times,
    terms=None,
    reference_X=None,
    reference_weight=None,
    offset=None,
    reference_offset=None,
    conditional_on=None,
    trajectory=None,
    reference_trajectory=None,
):
    """Evaluate reference predictors + one centered term before survival transform.

    Both engines supply predict_components and survival_from_predictors.
    Default terms have known features present in X, including interactions and
    categories; interval baseline terms are held at their reference values.
    Explicit requests for unavailable/contextual terms raise, rather than skip.
    Preparation does not claim PDP, causal effects or effect confidence intervals.
    """
    if not callable(getattr(estimator, "survival_from_predictors", None)):
        raise NotImplementedError(
            "Estimator has no supported survival predictor-evaluation adapter."
        )
    if trajectory is not None:
        return _trajectory_effects(
            estimator,
            X,
            trajectory,
            times=times,
            terms=terms,
            reference_X=reference_X,
            reference_trajectory=reference_trajectory,
            reference_weight=reference_weight,
            offset=offset,
            reference_offset=reference_offset,
            conditional_on=conditional_on,
        )
    if reference_trajectory is not None:
        raise ValueError("reference_trajectory requires a query trajectory.")
    frame = frame_for_estimator(estimator, X)
    current = _subject_components(
        estimator.predict_components(X, offset=offset), len(frame)
    )
    if reference_X is None and reference_offset is not None:
        raise ValueError("reference_offset requires reference_X.")
    reference = (
        None
        if reference_X is None
        else _subject_components(
            estimator.predict_components(reference_X, offset=reference_offset),
            len(reference_X),
        )
    )
    centered = center_additive_prediction(
        current, reference=reference, sample_weight=reference_weight
    )
    baseline = np.array(
        np.broadcast_to(centered.intercept, centered.link.shape), dtype=float
    )
    if centered.offset is not None:
        baseline += centered.offset
    prediction = estimator.survival_from_predictors(baseline)
    if not isinstance(prediction, SurvivalPrediction):
        raise TypeError("survival_from_predictors must return SurvivalPrediction.")
    grid = prediction._times(times)
    landmark = prediction._landmarks(conditional_on, np.inf)
    reference_survival = prediction.survival(grid, conditional_on=landmark)
    available = {}
    for term in centered.terms:
        metadata = centered.term_metadata.get(term)
        if (
            metadata is not None
            and metadata.kind in {"main", "interaction"}
            and metadata.features
            and any(feature in frame for feature in metadata.features)
            and all(
                feature in frame or feature == "survival_time"
                for feature in metadata.features
            )
        ):
            available[term] = tuple(metadata.features)
    requested = tuple(available) if terms is None else tuple(terms)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested terms must be unique.")
    if set(requested) - available.keys():
        raise NotImplementedError(
            f"Unavailable survival effect terms: {sorted(set(requested) - available.keys())}."
        )
    if not requested:
        raise ValueError("No feature terms are available for survival effects.")
    curves, effects, features, values = {}, {}, {}, {}
    for term in requested:
        curves[term] = estimator.survival_from_predictors(
            baseline + centered.terms[term]
        ).survival(grid, conditional_on=landmark)
        effects[term] = curves[term] - reference_survival
        features[term] = available[term]
        if "survival_time" in available[term]:
            endpoints = estimator.survival_time_grid()[1:]
            values[term] = np.stack(
                [
                    np.broadcast_to(endpoints, (len(frame), len(endpoints)))
                    if feature == "survival_time"
                    else np.broadcast_to(
                        frame[feature].to_numpy()[:, None], (len(frame), len(endpoints))
                    )
                    for feature in available[term]
                ],
                axis=-1,
            )
        else:
            values[term] = frame[list(available[term])].to_numpy()
    return SurvivalEffects(
        grid,
        reference_survival,
        curves,
        effects,
        values,
        features,
        centered.predictor_names,
        landmark,
        centered.backend,
    )


def _trajectory_effects(
    estimator,
    X,
    trajectory,
    *,
    times,
    terms,
    reference_X,
    reference_trajectory,
    reference_weight,
    offset,
    reference_offset,
    conditional_on,
):
    """Apply the existing reference-plus-term contract to supplied paths."""
    from ..survival.longitudinal import SurvivalTrajectory, _trajectory_layout

    if not isinstance(trajectory, SurvivalTrajectory) or len(trajectory) != len(X):
        raise ValueError("Query features must align with a SurvivalTrajectory.")
    if reference_X is None:
        if reference_trajectory is not None or reference_offset is not None:
            raise ValueError(
                "Explicit reference trajectory/offset requires reference_X."
            )
        reference_X, reference_trajectory, reference_offset = X, trajectory, offset
    if not isinstance(reference_trajectory, SurvivalTrajectory) or len(
        reference_X
    ) != len(reference_trajectory):
        raise ValueError(
            "Reference features must align with an explicit reference trajectory."
        )
    model_grid = estimator.survival_time_grid()
    cuts, _, _ = _trajectory_layout(model_grid, trajectory)
    cuts, _, _ = _trajectory_layout(model_grid, reference_trajectory, cuts=cuts)
    frame = frame_for_estimator(estimator, X)

    def components(data, path, known):
        raw = _subject_components(
            estimator.predict_components(data, offset=known), len(data)
        )
        _, rows, intervals = _trajectory_layout(model_grid, path, cuts=cuts)

        def select(value):
            return np.broadcast_to(value, raw.link.shape)[rows, intervals]

        return AdditivePrediction(
            link=select(raw.link),
            terms={name: select(value) for name, value in raw.terms.items()},
            intercept=select(raw.intercept),
            offset=None if raw.offset is None else select(raw.offset),
            backend=raw.backend,
            term_metadata=raw.term_metadata,
            predictor_names=tuple(
                f"log_hazard({a},{b}]" for a, b in zip(cuts[:-1], cuts[1:], strict=True)
            ),
        ), rows

    current, rows = components(X, trajectory, offset)
    reference, _ = components(reference_X, reference_trajectory, reference_offset)
    centered = center_additive_prediction(
        current, reference=reference, sample_weight=reference_weight
    )
    baseline = np.broadcast_to(centered.intercept, centered.link.shape).copy()
    if centered.offset is not None:
        baseline += centered.offset
    prediction = SurvivalPrediction(np.exp(baseline), cuts, backend=centered.backend)
    grid = prediction._times(times)
    landmark = prediction._landmarks(conditional_on, np.inf)
    reference_survival = prediction.survival(grid, conditional_on=landmark)
    available = {
        name: tuple(info.features)
        for name, info in centered.term_metadata.items()
        if info.kind in {"main", "interaction"}
        and any(f in frame for f in info.features)
        and all(f in frame or f == "survival_time" for f in info.features)
    }
    requested = tuple(available) if terms is None else tuple(terms)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select nonempty distinct trajectory effect terms.")
    if set(requested) - available.keys():
        raise NotImplementedError(
            "Requested trajectory effects are unavailable in the fitted term layout."
        )
    curves, effects, values = {}, {}, {}
    for name in requested:
        curves[name] = SurvivalPrediction(
            np.exp(baseline + centered.terms[name]), cuts, backend=centered.backend
        ).survival(grid, conditional_on=landmark)
        effects[name] = curves[name] - reference_survival
        values[name] = np.stack(
            [
                np.broadcast_to(cuts[1:], rows.shape)
                if feature == "survival_time"
                else frame[feature].to_numpy()[rows]
                for feature in available[name]
            ],
            axis=-1,
        )
    return SurvivalEffects(
        grid,
        reference_survival,
        curves,
        effects,
        values,
        {name: available[name] for name in requested},
        centered.predictor_names,
        landmark,
        centered.backend,
    )


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
    from ..survival.longitudinal import LongitudinalSurvivalTarget

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
