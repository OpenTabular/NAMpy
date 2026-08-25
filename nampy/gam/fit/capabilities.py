"""Fit backend capability and smoothing-method helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..coefficients import IdentityCoefficientTransform
from ..observations import IdentityObservationTransform


def coefficient_transform(model):
    compiled = getattr(getattr(model, "gam_result_", None), "compiled_model", None)
    transform = None if compiled is None else compiled.coefficient_transform
    if transform is not None:
        return transform
    return IdentityCoefficientTransform(0)


def observation_transform(model):
    compiled = getattr(getattr(model, "gam_result_", None), "compiled_model", None)
    transform = None if compiled is None else compiled.observation_transform
    if transform is not None:
        return transform
    size = int(getattr(model, "n_samples_", 0) or 0)
    return IdentityObservationTransform(size)


def has_transformed_coefficients(model) -> bool:
    return not coefficient_transform(model).is_identity


def has_transformed_observations(model) -> bool:
    return not observation_transform(model).is_identity


@dataclass(frozen=True)
class ModelFitCapabilities:
    """Derived model capabilities used by solver and criterion dispatch."""

    transformed_coefficients: bool
    transformed_observations: bool
    multiple_linear_predictors: bool
    closed_form_family: bool
    pirls_family: bool
    general_family: bool
    observation_transform_family_supported: bool


def model_fit_capabilities(model) -> ModelFitCapabilities:
    compiled = getattr(getattr(model, "gam_result_", None), "compiled_model", None)
    predictor_count = len(getattr(compiled, "predictors", ()) or ())
    family = getattr(model, "family", None)
    family_name = str(getattr(family, "name", "")).lower()
    link_name = str(getattr(family, "link_name", "")).lower()
    return ModelFitCapabilities(
        transformed_coefficients=has_transformed_coefficients(model),
        transformed_observations=has_transformed_observations(model),
        multiple_linear_predictors=predictor_count > 1,
        closed_form_family=bool(getattr(family, "supports_closed_form_solve", False)),
        pirls_family=bool(getattr(family, "supports_pirls", False)),
        general_family=str(getattr(family, "family_class", "")).lower() == "general",
        observation_transform_family_supported=(
            family_name == "gaussian" and link_name == "identity"
        ),
    )


def validate_observation_transform_support(model) -> None:
    """Reject observation transforms outside an implemented likelihood kernel."""
    capabilities = model_fit_capabilities(model)
    if (
        capabilities.transformed_observations
        and not capabilities.observation_transform_family_supported
    ):
        raise NotImplementedError(
            "Observation transforms currently require a Gaussian family with "
            "the identity link. Other likelihoods are rejected rather than "
            "fitted as if their observations were independent."
        )


def uses_closed_form_solver(model):
    return bool(getattr(model.family, "supports_closed_form_solve", False))


def needs_exact_gaussian_reparameterization(model):
    from .selection.reparam import can_use_exact_gaussian_ml_reml

    return (
        uses_closed_form_solver(model)
        and can_use_exact_gaussian_ml_reml(model)
        and any(
            bool(getattr(model.family, attr, False))
            for attr in ("supports_ml", "supports_reml", "supports_laml")
        )
    )


def raise_ml_reml_backend_error(model, method):
    from .selection.criteria.ml_reml import resolve_ml_reml_scoring_backend
    from .selection.reparam import can_use_simple_ml_reml_structure

    method = str(method).lower()
    capabilities = model_fit_capabilities(model)
    if capabilities.transformed_observations:
        if (
            capabilities.observation_transform_family_supported
            and method in {"ml", "reml", "laml"}
            and getattr(model.family, "known_scale", None) is None
        ):
            return
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} and an "
            "observation transform is outside the determinant-corrected "
            "Gaussian bam route. Use fixed smoothing or GCV."
        )
    if capabilities.transformed_coefficients:
        if capabilities.multiple_linear_predictors or capabilities.general_family:
            raise NotImplementedError(
                "Automatic smoothing for transformed multi-predictor/LSS models "
                "requires higher-order transformed Laplace derivatives that are "
                "not implemented. Fixed smoothing is supported."
            )
        raise NotImplementedError(
            "Automatic ML/REML/LAML smoothing is not implemented for transformed "
            "coefficients. Use fixed smoothing or the supported GCV/UBRE policy."
        )
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend is not None:
        return
    if not bool(getattr(model.family, f"supports_{method}", False)):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={model.family.name!r}."
        )
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            "currently available for this model configuration. "
            "The current ML/REML backend still rejects penalty layouts "
            "with null-space penalties coupling disconnected primary "
            "penalty components. Use 'fixed', 'gcv', or 'ubre' where "
            "available for those cases."
        )
    raise NotImplementedError(
        f"Automatic smoothing selection with method={method!r} is not "
        f"supported for family={model.family.name!r}."
    )


def coerce_general_family_smoothing_method(model, method, optimizer=None):
    """Mirror mgcv/R/mgcv.r::estimate.gam general.family method reset."""
    method = str(method).lower()
    optimizer = None if optimizer is None else str(optimizer).lower()
    family_class = str(getattr(model.family, "family_class", "")).lower()
    if family_class == "general" and (method != "reml" or optimizer == "efs"):
        return "reml"
    return method
