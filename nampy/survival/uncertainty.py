"""Joint survival curve/effect inference and separate replicated-fit variation."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .contracts import _readonly
from .interpretation import SurvivalEffects, prepare_survival_effects
from .prediction import SurvivalPrediction


@dataclass(frozen=True)
class SurvivalCurveInterval:
    """Curve/functional estimate and bounds (n,t), with explicit estimand metadata.

    For RMST, time contains restriction horizons; conditional_on contains the
    fixed landmarks and values are remaining lifetime in outcome time units.

    Coefficient intervals are approximate joint-Gaussian simulation
    intervals; bootstrap intervals use subject refits with distinct source labels.
    Between-fit ranges are empirical variability, not confidence
    intervals. Finite-grid simultaneous coverage is explicitly labeled by scope;
    it does not imply continuous-time or population-wide coverage. Standard
    errors are on the declared transformation scale. Pre-landmark cells are NaN.
    """

    time: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    conditional_on: np.ndarray
    level: float
    n_samples: int
    uncertainty_kind: str
    coverage: str
    standard_error: np.ndarray | None = None
    critical_value: float | None = None
    transformation: str = "survival"
    scope: str = "pointwise"
    estimand: str = "survival"

    def __post_init__(self):
        for name in ("time", "estimate", "lower", "upper", "conditional_on"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.standard_error is not None:
            object.__setattr__(self, "standard_error", _readonly(self.standard_error))


def _simultaneous_limits(point, samples, valid, level):
    """Max absolute standardized deviations, jointly over every valid cell.

    The nonlinear draw SD (ddof=1) supplies the scale. Quantiles use R type 8,
    matching the existing gratia-reference joint-functional calibration.
    """
    if not valid.any():
        raise ValueError("Simultaneous inference needs at least one valid grid cell.")
    values, center = samples[:, valid], point[valid]
    if not np.isfinite(values).all() or not np.isfinite(center).all():
        raise FloatingPointError("Simultaneous calibration requires finite transformed draws and estimates.")
    # Translation and scaling preserve the sample SD while avoiding squaring
    # tiny tail contrasts (or rounding a constant nonzero column's variance).
    differences = values-values[0]
    magnitude = np.max(np.abs(differences), axis=0)
    normalized = np.divide(differences, magnitude, out=np.zeros_like(differences), where=magnitude > 0)
    scale = np.std(normalized, axis=0, ddof=1)*magnitude
    active = scale > 0
    if np.any(values[:, ~active] != center[~active]):
        raise ValueError("Zero-variance simulation cells must equal the fitted estimate.")
    critical = (float(np.quantile(np.max(np.abs(
        (values[:, active]-center[active])/scale[active]), axis=1),
        level, method="median_unbiased")) if active.any() else 0.0)
    lower = np.full(point.shape, np.nan)
    upper, standard_error = lower.copy(), lower.copy()
    lower[valid], upper[valid] = center-critical*scale, center+critical*scale
    standard_error[valid] = scale
    return lower, upper, standard_error, critical


def _curve_interval(predictions, times, *, point_prediction, level, conditional_on,
                    uncertainty_kind, coverage, interval="pointwise"):
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie in (0,1).")
    if interval not in {"pointwise", "simultaneous"}:
        raise ValueError("interval must be 'pointwise' or 'simultaneous'.")
    items = tuple(predictions)
    if len(items) < 2 or not all(isinstance(item, SurvivalPrediction) for item in items):
        raise ValueError("At least two SurvivalPrediction snapshots are required.")
    n = items[0].n_subjects
    if any(item.n_subjects != n for item in items):
        raise ValueError("All predictions must have identical subject rows and order.")
    grid = items[0]._times(times)
    landmark = items[0]._landmarks(conditional_on, np.inf)
    curves = np.stack([item.survival(grid, conditional_on=landmark) for item in items])
    valid = grid[None, :] >= landmark[:, None]
    if not np.isfinite(curves[:, valid]).all():
        raise FloatingPointError("Interval calculation requires finite valid survival cells.")
    lower = np.full((n, len(grid)), np.nan)
    upper = lower.copy()
    lower[valid], upper[valid] = np.quantile(curves[:, valid], [(1-level)/2, (1+level)/2], axis=0)
    if point_prediction is None:
        estimate = np.mean(curves, axis=0)
    else:
        if point_prediction.n_subjects != n:
            raise ValueError("Point prediction must align with sampled subjects.")
        estimate = point_prediction.survival(grid, conditional_on=landmark)
        if not np.isfinite(estimate[valid]).all():
            raise FloatingPointError("Point survival must be finite on valid cells.")
    if interval == "simultaneous":
        if point_prediction is None or uncertainty_kind == "between_fits":
            raise NotImplementedError("Simultaneous bands require an original fitted estimate and coefficient or subject-bootstrap draws; generic replicated fits are unsupported.")
        hazards = -np.stack([item.log_survival(grid, conditional_on=landmark) for item in items])
        point_hazard = -point_prediction.log_survival(grid, conditional_on=landmark)
        if (not np.isfinite(hazards[:, valid]).all() or not np.isfinite(point_hazard[valid]).all()
                or np.any(hazards[:, valid] < 0) or np.any(point_hazard[valid] < 0)):
            raise FloatingPointError("Simultaneous survival bands require finite nonnegative cumulative hazards.")
        positive = valid & (point_hazard > 0)
        if np.any(hazards[:, valid & ~positive] != 0) or np.any(hazards[:, positive] <= 0):
            raise FloatingPointError("Cumulative-hazard zero boundaries must agree in every joint draw.")
        transformed = np.zeros_like(hazards)
        point = np.zeros_like(point_hazard)
        transformed[:, positive] = np.log(hazards[:, positive])
        point[positive] = np.log(point_hazard[positive])
        lo, hi, se, critical = _simultaneous_limits(point, transformed, valid, level)
        with np.errstate(over="ignore", under="ignore"):
            lower[positive], upper[positive] = np.exp(-np.exp(hi[positive])), np.exp(-np.exp(lo[positive]))
        lower[valid & ~positive], upper[valid & ~positive] = 1, 1
        return SurvivalCurveInterval(grid, estimate, lower, upper, landmark, level,
            len(items), uncertainty_kind, "simultaneous", se, critical,
            "log_cumulative_hazard", "all_subjects_and_times")
    return SurvivalCurveInterval(grid, estimate, lower, upper, landmark, level,
                                 len(items), uncertainty_kind, coverage)


def combine_survival_predictions(predictions, *, times, level=.95, conditional_on=None):
    """Average response-level survival and report central between-fit quantiles.

    Supply independently fitted models' snapshots on identical subject rows.
    This function does not infer replication, align/reorder rows, pool hazards
    or parameters, or infer coefficient uncertainty from a single neural fit.
    """
    return _curve_interval(predictions, times, point_prediction=None, level=level,
                           conditional_on=conditional_on, uncertainty_kind="between_fits",
                           coverage="empirical_range")


def prepare_survival_curve_interval(
    estimator, X, *, times, offset=None, conditional_on=None, level=.95,
    covariance="bayes", n_draws=1000, seed=None, standard_draws=None, interval="pointwise",
):
    """Statistical joint coefficient intervals, pointwise or finite-grid simultaneous.

    Bayesian covariance is conditional on smoothing; 'unconditional' requires
    available smoothing-corrected covariance. Other covariance choices and
    single-neural-fit coefficient intervals are unsupported. Simulation belongs
    to the backend; probability transformation and quantiles are shared here.
    Simultaneous coverage is joint over all supplied valid subject/time cells,
    using max standardized deviations of log cumulative hazard. It is not a
    continuous-time band. Single neural fits have no supported coefficient route.
    """
    if covariance not in {"bayes", "unconditional"}:
        raise NotImplementedError("Survival intervals support bayes or available unconditional covariance only.")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie in (0,1).")
    if interval not in {"pointwise", "simultaneous"}:
        raise ValueError("interval must be 'pointwise' or 'simultaneous'.")
    if not callable(getattr(estimator, "sample_survival_predictions", None)):
        raise NotImplementedError("Estimator has no supported survival coefficient-simulation adapter.")
    prediction = estimator.predict_distribution(X, offset=offset)
    prediction._times(times)
    samples = estimator.sample_survival_predictions(
        X, offset=offset, n_draws=n_draws, covariance=covariance,
        seed=seed, standard_draws=standard_draws)
    return _curve_interval(samples, times, point_prediction=prediction, level=level,
                           conditional_on=conditional_on,
                           uncertainty_kind=("coefficient_conditional" if covariance == "bayes"
                                             else "coefficient_smoothing_corrected"),
                           coverage="pointwise", interval=interval)


@dataclass(frozen=True)
class SurvivalEffectSamples:
    """Paired joint-draw predictions for reference-plus-term contrasts."""

    reference: tuple
    terms: dict

    def __post_init__(self):
        if len(self.reference) < 2 or not self.terms:
            raise ValueError("At least two paired effect draws and one term are required.")
        n = self.reference[0].n_subjects
        for values in (self.reference, *self.terms.values()):
            if len(values) != len(self.reference) or any(
                    not isinstance(value, SurvivalPrediction) or value.n_subjects != n for value in values):
                raise ValueError("Effect predictions must share draw counts, subject rows and order.")
        object.__setattr__(self, "reference", tuple(self.reference))
        object.__setattr__(self, "terms", MappingProxyType({key: tuple(value) for key, value in self.terms.items()}))


@dataclass(frozen=True)
class SurvivalEffectInterval:
    """Prepared effect contrasts plus intervals, retaining term/feature metadata."""

    effects: SurvivalEffects
    intervals: dict

    def __post_init__(self):
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))


def prepare_survival_effect_interval(
    estimator, X, *, times, terms=None, reference_X=None, reference_weight=None,
    offset=None, reference_offset=None, conditional_on=None, level=.95,
    covariance="bayes", n_draws=1000, seed=None, standard_draws=None, interval="pointwise",
):
    """Paired survival effect-contrast intervals, preserving shared coefficient draws.

    The contrast is the existing reference-plus-centered-term survival minus
    reference survival. Fixed reference rows/weights are recentered in every
    draw, retaining their coefficient covariance. No reference-population
    resampling is implied. Simultaneous coverage includes all selected terms,
    subjects and valid times jointly, on the survival-difference scale.
    """
    if covariance not in {"bayes", "unconditional"}:
        raise NotImplementedError("Survival intervals support bayes or available unconditional covariance only.")
    if interval not in {"pointwise", "simultaneous"} or not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("Select pointwise/simultaneous intervals and level in (0,1).")
    if not callable(getattr(estimator, "sample_survival_effect_predictions", None)):
        raise NotImplementedError("Estimator has no supported joint survival effect-simulation adapter.")
    prepared = prepare_survival_effects(estimator, X, times=times, terms=terms,
        reference_X=reference_X, reference_weight=reference_weight,
        offset=offset, reference_offset=reference_offset, conditional_on=conditional_on)
    names = tuple(prepared.effects)
    sampled = estimator.sample_survival_effect_predictions(X, terms=names,
        reference_X=reference_X, reference_weight=reference_weight,
        offset=offset, reference_offset=reference_offset, covariance=covariance,
        n_draws=n_draws, seed=seed, standard_draws=standard_draws)
    if not isinstance(sampled, SurvivalEffectSamples) or tuple(sampled.terms) != names:
        raise ValueError("Joint effect samples must match the prepared ordered terms.")
    reference = np.stack([value.survival(prepared.time, conditional_on=prepared.conditional_on)
                          for value in sampled.reference])
    curves = np.stack([np.stack([value.survival(prepared.time, conditional_on=prepared.conditional_on)
                                for value in sampled.terms[name]])-reference for name in names], axis=1)
    point = np.stack([prepared.effects[name] for name in names])
    valid = np.broadcast_to(prepared.time[None, None, :] >= prepared.conditional_on[None, :, None], point.shape)
    if not np.isfinite(curves[:, valid]).all() or not np.isfinite(point[valid]).all():
        raise FloatingPointError("Effect intervals require finite valid survival contrasts.")
    if interval == "simultaneous":
        lower, upper, se, critical = _simultaneous_limits(point, curves, valid, level)
        lower, upper = np.maximum(lower, -1), np.minimum(upper, 1)
    else:
        lower = np.full(point.shape, np.nan)
        upper = lower.copy()
        lower[valid], upper[valid] = np.quantile(curves[:, valid], [(1-level)/2, (1+level)/2], axis=0)
        se, critical = None, None
    kind = "coefficient_conditional" if covariance == "bayes" else "coefficient_smoothing_corrected"
    return SurvivalEffectInterval(prepared, {name: SurvivalCurveInterval(
        prepared.time, point[i], lower[i], upper[i], prepared.conditional_on, level,
        len(sampled.reference), kind, interval, None if se is None else se[i], critical,
        "survival_difference", "all_selected_terms_subjects_and_times" if interval == "simultaneous" else "pointwise",
        "reference_plus_term_survival_difference") for i, name in enumerate(names)})


def _functional_values(prediction, grid, landmark, kind):
    valid = grid[None, :] >= landmark[:, None]
    if kind == "hazard":
        values = prediction.hazard(grid)
        values[~valid] = np.nan
        return values
    values = np.full((prediction.n_subjects, len(grid)), np.nan)
    for j, horizon in enumerate(grid):
        rows = np.flatnonzero(valid[:, j])
        if len(rows):
            values[rows, j] = prediction.take(rows).rmst(horizon, conditional_on=landmark[rows])
    return values


def _prepare_functional_interval(
    estimator, X, *, grid, kind, offset, conditional_on, level, covariance,
    n_draws, seed, standard_draws, interval,
):
    """Transport the same joint coefficient snapshots through a shared functional."""
    if covariance not in {"bayes", "unconditional"}:
        raise NotImplementedError("Survival intervals support bayes or available unconditional covariance only.")
    if interval not in {"pointwise", "simultaneous"} or not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("Select pointwise/simultaneous intervals and level in (0,1).")
    if not callable(getattr(estimator, "sample_survival_predictions", None)):
        raise NotImplementedError("Estimator has no supported survival coefficient-simulation adapter.")
    prediction = estimator.predict_distribution(X, offset=offset)
    grid = prediction._times(grid)
    landmark = prediction._landmarks(conditional_on, np.inf)
    if np.any(landmark > prediction.support[1]):
        raise NotImplementedError("Landmark lies beyond supported cuts.")
    if kind == "hazard" and prediction.distribution is not None and np.any(grid == 0):
        raise NotImplementedError("Parametric hazard limits at zero are not exposed.")
    samples = tuple(estimator.sample_survival_predictions(X, offset=offset,
        covariance=covariance, n_draws=n_draws, seed=seed, standard_draws=standard_draws))
    kind_label = "coefficient_conditional" if covariance == "bayes" else "coefficient_smoothing_corrected"
    return _functional_interval(prediction, samples, grid, landmark, kind, level, interval, kind_label)


def _functional_interval(prediction, samples, grid, landmark, kind, level, interval, kind_label):
    """Shared functional transport for joint coefficient or subject-refit snapshots."""
    if len(samples) < 2 or any(not isinstance(value, SurvivalPrediction)
                              or value.n_subjects != prediction.n_subjects for value in samples):
        raise ValueError("At least two joint predictions aligned with subject rows are required.")
    point = _functional_values(prediction, grid, landmark, kind)
    draws = np.stack([_functional_values(value, grid, landmark, kind) for value in samples])
    valid = grid[None, :] >= landmark[:, None]
    if (not np.isfinite(point[valid]).all() or not np.isfinite(draws[:, valid]).all()
            or np.any(point[valid] < 0) or np.any(draws[:, valid] < 0)):
        raise FloatingPointError("Functional intervals require finite nonnegative valid estimates and draws.")
    se, critical = None, None
    transformation = kind
    if interval == "pointwise":
        lower = np.full(point.shape, np.nan)
        upper = lower.copy()
        lower[valid], upper[valid] = np.quantile(draws[:, valid], [(1-level)/2, (1+level)/2], axis=0)
    elif kind == "hazard":
        positive = valid & (point > 0)
        if np.any(draws[:, valid & ~positive] != 0) or np.any(draws[:, positive] <= 0):
            raise FloatingPointError("Hazard zero boundaries must agree in every joint draw.")
        transformed = np.zeros_like(draws)
        center = np.zeros_like(point)
        transformed[:, positive], center[positive] = np.log(draws[:, positive]), np.log(point[positive])
        lo, hi, se, critical = _simultaneous_limits(center, transformed, valid, level)
        lower, upper = np.full(point.shape, np.nan), np.full(point.shape, np.nan)
        with np.errstate(over="ignore", under="ignore"):
            lower[positive], upper[positive] = np.exp(lo[positive]), np.exp(hi[positive])
        lower[valid & ~positive], upper[valid & ~positive] = 0, 0
        if not np.isfinite(lower[valid]).all() or not np.isfinite(upper[valid]).all():
            raise FloatingPointError("Hazard interval bounds exceed finite numerical range.")
        transformation = "log_hazard"
    else:
        lower, upper, se, critical = _simultaneous_limits(point, draws, valid, level)
    if kind == "rmst":
        width = np.maximum(grid[None, :]-landmark[:, None], 0)
        lower = np.minimum(np.maximum(lower, 0), width)
        upper = np.minimum(np.maximum(upper, 0), width)
    estimand = "hazard" if kind == "hazard" else "restricted_mean_remaining_lifetime"
    return SurvivalCurveInterval(grid, point, lower, upper, landmark, level, len(samples),
        kind_label, interval, se, critical, transformation,
        ("all_subjects_and_times" if kind == "hazard" else "all_subjects_and_horizons")
        if interval == "simultaneous" else "pointwise", estimand)


def prepare_survival_hazard_interval(
    estimator, X, *, times, offset=None, conditional_on=None, level=.95,
    covariance="bayes", n_draws=1000, seed=None, standard_draws=None, interval="pointwise",
):
    """Hazard coefficient intervals from shared joint prediction snapshots.

    Pointwise limits are central draw quantiles. Simultaneous bands calibrate
    log hazard jointly over every valid supplied subject/time cell. Conditioning
    masks pre-landmark cells but does not alter the hazard at/after a landmark.
    Piecewise hazards retain (left,right] cut semantics. Parametric zero-time
    limits and single-neural-fit coefficient inference remain unsupported.
    """
    return _prepare_functional_interval(estimator, X, grid=times, kind="hazard",
        offset=offset, conditional_on=conditional_on, level=level, covariance=covariance,
        n_draws=n_draws, seed=seed, standard_draws=standard_draws, interval=interval)


def prepare_survival_rmst_interval(
    estimator, X, *, horizons, offset=None, conditional_on=None, level=.95,
    covariance="bayes", n_draws=1000, seed=None, standard_draws=None, interval="pointwise",
):
    """RMST coefficient intervals; integrate each joint draw, not survival limits.

    Horizons are a strictly increasing finite nonnegative vector. Results have
    shape (subjects,horizons); time stores horizons. Conditional remaining RMST
    integrates from each fixed landmark to its horizon. Earlier horizons are
    NaN; equal landmarks give zero. Simultaneous coverage is joint over every
    valid subject/horizon, in time units, intersected with [0,horizon-landmark].
    Existing exact piecewise / controlled parametric integration is reused for
    every draw, so parametric cost grows with draws, subjects and horizons.
    Single neural fits have no supported coefficient interval route.
    """
    return _prepare_functional_interval(estimator, X, grid=horizons, kind="rmst",
        offset=offset, conditional_on=conditional_on, level=level, covariance=covariance,
        n_draws=n_draws, seed=seed, standard_draws=standard_draws, interval=interval)
