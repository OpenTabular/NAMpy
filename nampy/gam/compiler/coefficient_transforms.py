"""Compiler ownership helpers for coefficient-transform layouts."""

from __future__ import annotations

import numpy as np

from ..coefficients import (
    CoordinatewiseCoefficientTransform,
    IdentityCoefficientTransform,
    compose_coefficient_transforms,
)


def _configured_transform(
    mask,
    *,
    positive_map,
    softplus_beta,
    softplus_threshold,
    covariance_transport="jacobian",
):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return IdentityCoefficientTransform(mask.size)
    return CoordinatewiseCoefficientTransform(
        mask,
        positive_map=positive_map,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        covariance_transport=covariance_transport,
    )


def configure_compiled_coefficient_transforms(
    compiled_model,
    *,
    positive_map: str = "exp",
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    """Attach one canonical transform at term, predictor, and model levels.

    Runtime terms declare *which* coordinates are constrained.  Fit/model
    configuration declares *how* positive coordinates are parameterized.
    """
    predictor_blocks = []
    for predictor in compiled_model.predictors:
        term_blocks = []
        for term in predictor.compiled_terms:
            covariance_transport = dict(term.metadata or {}).get(
                "coefficient_covariance_transport", "jacobian"
            )
            term.coefficient_transform = _configured_transform(
                term.positive_coefficient_mask,
                positive_map=positive_map,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                covariance_transport=covariance_transport,
            )
            term_blocks.append(term.coefficient_transform)
        predictor.coefficient_transform = compose_coefficient_transforms(term_blocks)
        expected = int(predictor.n_coef)
        if predictor.coefficient_transform.size != expected:
            raise RuntimeError(
                f"Predictor {predictor.name!r} transform size "
                f"{predictor.coefficient_transform.size} does not match {expected}."
            )
        full_blocks = []
        if predictor.has_intercept:
            full_blocks.append(IdentityCoefficientTransform(1))
        full_blocks.append(predictor.coefficient_transform)
        predictor_blocks.append(compose_coefficient_transforms(full_blocks))

    for term in compiled_model.compiled_terms:
        covariance_transport = dict(term.metadata or {}).get(
            "coefficient_covariance_transport", "jacobian"
        )
        term.coefficient_transform = _configured_transform(
            term.positive_coefficient_mask,
            positive_map=positive_map,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            covariance_transport=covariance_transport,
        )

    compiled_model.coefficient_transform = compose_coefficient_transforms(
        predictor_blocks
    )
    expected_full = int(
        sum(pred.n_coef + int(bool(pred.has_intercept)) for pred in compiled_model.predictors)
    )
    if compiled_model.coefficient_transform.size != expected_full:
        raise RuntimeError(
            "Compiled-model transform width does not match its full coefficient layout."
        )
    compiled_model.positive_coefficient_mask = np.asarray(
        getattr(compiled_model.coefficient_transform, "positive_mask", np.zeros(expected_full)),
        dtype=bool,
    )
    return compiled_model


__all__ = ["configure_compiled_coefficient_transforms"]
