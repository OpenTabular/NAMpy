"""Registry-owned statistical adapters to shared probability predictions."""

import numpy as np

from ...distributions import DistributionPrediction
from ...distributions.mixtures import (
    inflated_beta_prediction,
    zero_adjusted_gamma_prediction,
)


def _parameters(family, parameters):
    parameters = np.asarray(parameters, dtype=np.float64)
    width = len(
        getattr(family, "distribution_parameter_names", ()) or family.parameter_names
    )
    if parameters.ndim != 2 or parameters.shape[1] != width:
        raise ValueError(
            f"Expected statistical natural parameters with shape (n, {width})."
        )
    missing = np.all(np.isnan(parameters), axis=1)
    if np.any(~np.isfinite(parameters[~missing])):
        raise ValueError(
            "Parameters must be finite or all NaN in a missing prediction row."
        )
    return parameters, missing


def direct_prediction(family, parameters, *, law_factory, context=None):
    parameters, _ = _parameters(family, parameters)
    law = law_factory()
    return DistributionPrediction(
        law,
        dict(zip(law.spec.parameter_names, parameters.T, strict=True)),
        context={} if context is None else context,
    )


def observation_prediction(family, parameters, *, law_factory=None, context=None):
    return family._distribution(parameters)


def inflated_beta_adapter(family, parameters, *, law_factory=None, context=None):
    parameters, _ = _parameters(family, parameters)
    return inflated_beta_prediction(parameters)


def zero_adjusted_gamma_adapter(family, parameters, *, law_factory=None, context=None):
    parameters, _ = _parameters(family, parameters)
    return zero_adjusted_gamma_prediction(parameters)


def student_prediction(family, parameters, *, law_factory, context=None):
    parameters, missing = _parameters(family, parameters)
    return DistributionPrediction(
        law_factory(),
        {
            "df": np.where(missing, np.nan, family.df),
            "loc": parameters[:, 0],
            "scale": parameters[:, 1],
        },
    )


def beta_prediction(family, parameters, *, law_factory, context=None):
    parameters, _ = _parameters(family, parameters)
    mu, phi = parameters.T
    if np.any((mu <= 0) | (mu >= 1)) or np.any(phi <= 0):
        raise ValueError("Beta mu must be inside (0,1) and phi must be positive.")
    return DistributionPrediction(
        law_factory(), {"alpha": mu * phi, "beta": (1 - mu) * phi}
    )


def gamma_prediction(family, parameters, *, law_factory, context=None):
    parameters, _ = _parameters(family, parameters)
    mu, sigma = parameters.T
    if np.any(mu <= 0) or np.any(sigma <= 0):
        raise ValueError("Gamma mu and sigma must be positive.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        shape = 1.0 / np.square(sigma)
        rate = shape / mu
    return DistributionPrediction(law_factory(), {"shape": shape, "rate": rate})


def distribution_prediction(family, parameters, *, context=None):
    """Use the fitting registry's exact-type adapter and context declaration."""
    from ..families.registry import GENERAL_FAMILIES, family_spec_for_instance

    spec = family_spec_for_instance(family)
    if spec is None or not spec.supports_distribution_prediction:
        supported = ", ".join(
            name
            for name, item in GENERAL_FAMILIES.items()
            if item.supports_distribution_prediction
        )
        raise NotImplementedError(
            f"predict_distribution has no statistical adapter for {type(family).__name__}. "
            f"Supported families are {supported}; existing predict/score methods remain available."
        )
    if context and not spec.prediction_context_names:
        raise ValueError(f"{family.name} does not accept observation context.")
    adapter = spec.prediction_adapter or direct_prediction
    return adapter(family, parameters, law_factory=spec.law_factory, context=context)
