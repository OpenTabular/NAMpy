"""Statistical survival simulation in the existing joint fit coordinates."""

import numpy as np

from ..fit.capabilities import has_transformed_coefficients
from .functionals import coefficient_draws


def validate_coefficient_coordinates(model):
    """Validate the fitted coordinate mapping without drawing or evaluating rows."""
    beta = model.gam_result_.fit_core_solution.fit_result.coef_full
    compiled = model.gam_result_.require_compiled_model()
    mapping = compiled.fit_to_prediction_parameterization_map
    if has_transformed_coefficients(model) or (
        mapping is not None and not np.array_equal(mapping, np.eye(len(beta)))
    ):
        raise NotImplementedError(
            "Survival coefficient simulation for transformed or wider prediction coordinates is unsupported."
        )
    return compiled, beta


def coefficient_design(model, matrix, *, n_rows, n_predictors, piecewise):
    """Map a compiled predictor matrix to the joint coefficient sampler."""
    compiled, beta = validate_coefficient_coordinates(model)
    matrix = np.asarray(matrix)
    if piecewise:
        design = matrix.reshape(n_rows, n_predictors, matrix.shape[1])
    else:
        design = np.zeros((n_rows, n_predictors, matrix.shape[1]))
        for k, columns in enumerate(compiled.predictor_full_indices):
            design[:, k, columns] = matrix[:, columns]
    if design.shape[-1] != len(beta) or not np.isfinite(design).all():
        raise NotImplementedError(
            "Survival coefficient simulation requires aligned finite prediction coordinates."
        )
    return design, beta


def survival_coefficient_draws(model, *, covariance, n_draws, seed, standard_draws):
    """Use the statistical joint sampler with declared survival covariance routes."""
    if covariance not in {"bayes", "unconditional"}:
        raise NotImplementedError(
            "Survival simulation supports bayes or unconditional covariance only."
        )
    draws = coefficient_draws(
        model,
        n_draws=n_draws,
        seed=seed,
        covariance=covariance,
        standard_draws=standard_draws,
    )
    if len(draws) < 2:
        raise ValueError("Survival intervals require at least two coefficient draws.")
    return draws
