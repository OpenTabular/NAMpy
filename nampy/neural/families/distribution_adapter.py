"""Neural natural-parameter adapters selected by the family registry."""

import numpy as np
import torch

from ...distributions import DistributionPrediction
from ...distributions.mixtures import (
    inflated_beta_prediction,
    zero_adjusted_gamma_prediction,
)


def direct_prediction(family, parameters, *, law_factory):
    """Map matching natural columns to a shared law without changing transforms."""
    law = law_factory()
    parameters = np.asarray(parameters, dtype=np.float64)
    width = len(law.spec.parameter_names)
    if parameters.ndim != 2 or parameters.shape[1] != width:
        raise ValueError(f"Expected neural natural parameters with shape (n, {width}).")
    missing = np.all(np.isnan(parameters), axis=1)
    if np.any(~np.isfinite(parameters[~missing])):
        raise ValueError(
            "Parameters must be finite or all NaN in a missing prediction row."
        )
    context = (
        {"trials": np.full(len(parameters), family.trials)}
        if law.spec.context_names == ("trials",)
        else {}
    )
    return DistributionPrediction(
        law,
        dict(zip(law.spec.parameter_names, parameters.T, strict=True)),
        context=context,
    )


def negative_binomial_prediction(family, parameters, *, law_factory):
    prediction = direct_prediction(family, parameters, law_factory=law_factory)
    parameters = np.asarray(parameters, dtype=np.float64)
    missing = np.all(np.isnan(parameters), axis=1)
    # Preserve the existing guard against a different law in clamped training regimes.
    values = torch.as_tensor(parameters[~missing], dtype=torch.float32)
    mean, alpha = values.T
    size = 1 / torch.clamp(alpha, min=family.eps)
    denominator = mean + size
    probability = mean / torch.clamp(denominator, min=family.eps)
    clipped = torch.clamp(probability, min=family.eps, max=1 - family.eps)
    if torch.any(
        (alpha < family.eps)
        | (denominator < family.eps)
        | (probability != clipped)
        | (probability <= 0)
        | (probability >= 1)
    ):
        raise NotImplementedError(
            "predict_distribution does not support the clamped neural negative-binomial "
            "parameter regime; training and score retain their existing clipping behavior."
        )
    return prediction


def observation_prediction(family, parameters, *, law_factory=None):
    return family.distribution_prediction(parameters)


def inflated_beta_adapter(family, parameters, *, law_factory=None):
    return inflated_beta_prediction(parameters)


def zero_adjusted_gamma_adapter(family, parameters, *, law_factory=None):
    return zero_adjusted_gamma_prediction(parameters)


def distribution_prediction(family, parameters):
    """Resolve an exact registered training family, preserving custom transforms."""
    from .registry import FAMILY_REGISTRY, family_spec_for_instance

    spec = family_spec_for_instance(family)
    if spec is None or not spec.supports_distribution_prediction:
        supported = ", ".join(
            name
            for name, item in FAMILY_REGISTRY.items()
            if item.supports_distribution_prediction
        )
        raise NotImplementedError(
            f"predict_distribution has no neural adapter for {type(family).__name__}. "
            f"Supported families are {supported}; existing predict/score methods remain available."
        )
    adapter = spec.prediction_adapter or direct_prediction
    return adapter(family, parameters, law_factory=spec.law_factory)
