"""Observed-event likelihoods; latent event-time laws retain their own owner."""

import numpy as np

from ..distributions import DistributionPrediction, LogNormalLaw, WeibullLaw
from .contracts import SurvivalTarget, validate_weights


def log_entry_ratio(time, entry):
    """log(entry/time) for positive entry, stable near equality and near zero."""
    time, entry = np.asarray(time), np.asarray(entry)
    near = entry > 0.5 * time
    result = np.empty_like(time, dtype=float)
    result[near] = np.log1p((entry[near] - time[near]) / time[near])
    result[~near] = np.log(entry[~near]) - np.log(time[~near])
    return result


def parametric_distribution(likelihood, parameters):
    """Adapt natural parameters to existing scalar laws, without censoring them."""
    laws = {"weibull": WeibullLaw, "lognormal": LogNormalLaw}
    if likelihood not in laws:
        raise NotImplementedError(f"No parametric survival law for {likelihood!r}.")
    law = laws[likelihood]()
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("Natural survival parameters must have shape (n_subjects, 2).")
    return DistributionPrediction(
        law, dict(zip(law.spec.parameter_names, parameters.T, strict=True))
    )


def observed_log_likelihood(target, distribution):
    """Right-censored likelihood conditional on entry; excludes censoring density.

    This is not the density of censored times. Weibull cumulative-hazard
    increments use expm1 to retain close entry/exit differences.
    """
    if not isinstance(target, SurvivalTarget):
        raise TypeError("target must be SurvivalTarget.")
    if distribution.spec.name not in {"weibull", "lognormal"}:
        raise NotImplementedError(
            "Only Weibull/lognormal observed likelihoods are supported."
        )
    if len(next(iter(distribution.parameters.values()))) != len(target):
        raise ValueError("Target and distribution subjects must align.")
    if distribution.spec.name == "weibull":
        scale, shape = (distribution.parameters[name] for name in ("scale", "shape"))
        log_t = np.log(target.time)
        u = shape * (log_t - np.log(scale))
        fraction = np.ones(len(target))
        positive = target.entry > 0
        # log(entry/time) without subtracting two nearly equal logarithms.
        ratio = log_entry_ratio(target.time[positive], target.entry[positive])
        fraction[positive] = -np.expm1(shape[positive] * ratio)
        return target.event * (np.log(shape) - log_t + u) - np.exp(u) * fraction
    result = distribution.log_survival(target.time)
    if target.event.any():
        # Slice the law so unused exact-event density branches are not evaluated.
        event = target.event
        exact = DistributionPrediction(
            distribution.law,
            {name: values[event] for name, values in distribution.parameters.items()},
        )
        result[event] = exact.log_prob(target.time[event])
    positive = target.entry > 0
    if positive.any():
        entered = DistributionPrediction(
            distribution.law,
            {
                name: values[positive]
                for name, values in distribution.parameters.items()
            },
        )
        result[positive] -= entered.log_survival(target.entry[positive])
    return result


def validate_parametric_fit(target, sample_weight=None):
    """First-release estimability guard, not a guarantee for arbitrary designs."""
    weights = validate_weights(target, sample_weight, require_event=True)
    events = target.time[target.event & (weights > 0)]
    if len(np.unique(events)) < 2:
        raise NotImplementedError(
            "Parametric survival fitting requires at least two distinct positive-weight exact events; "
            "degenerate and boundary fits are unsupported."
        )
    return weights


def parameter_offsets(offset, n_subjects):
    """Canonical two-predictor offset matrix shared by both survival backends."""
    if offset is None:
        return np.zeros((n_subjects, 2))
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (n_subjects, 2) or not np.isfinite(offset).all():
        raise ValueError(
            "Parametric offsets must be finite with shape (n_subjects, 2)."
        )
    return offset.copy()
