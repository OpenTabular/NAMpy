"""Fit/public parameterization contracts and export helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .._model_state import _coef_column_offset, _compiled_model, _fit_intercept
from ..linalg import symmetrize_matrix

FIT_PARAMETER_SPACE = "fit"
PREDICTION_PARAMETER_SPACE = "prediction"


def normalize_parameter_space(space: str | None) -> str:
    value = str(space or FIT_PARAMETER_SPACE).strip().lower()
    if value not in {FIT_PARAMETER_SPACE, PREDICTION_PARAMETER_SPACE}:
        raise ValueError(f"Unknown parameterization space {space!r}.")
    return value


def prediction_parameterization_map(model) -> np.ndarray | None:
    compiled_model = _compiled_model(model)
    if compiled_model is None:
        return None
    direct = getattr(compiled_model, "fit_to_prediction_parameterization_map", None)
    if direct is not None:
        return np.asarray(direct, dtype=np.float64)
    metadata = dict(getattr(compiled_model, "metadata", {}) or {})
    legacy = metadata.get("fit_to_prediction_parameterization_map", None)
    if legacy is None:
        return None
    return np.asarray(legacy, dtype=np.float64)


def _transform_covariance_to_prediction_space(
    cov: np.ndarray | None,
    P: np.ndarray | None,
    *,
    source_space: str | None,
) -> tuple[np.ndarray | None, str]:
    target_space = PREDICTION_PARAMETER_SPACE
    if cov is None:
        return None, target_space
    space = normalize_parameter_space(source_space)
    arr = np.asarray(cov, dtype=np.float64)
    if space == PREDICTION_PARAMETER_SPACE or P is None:
        return arr.copy(), target_space
    return symmetrize_matrix(P @ arr @ P.T), target_space


def export_fit_result_to_prediction_space(model, fit_result):
    P = prediction_parameterization_map(model)

    coef_space = normalize_parameter_space(getattr(fit_result, "coef_space", None))
    coef_full = np.asarray(fit_result.coef_full, dtype=np.float64)
    if coef_space == FIT_PARAMETER_SPACE and P is not None:
        coef_full = np.asarray(P @ coef_full, dtype=np.float64)
    else:
        coef_full = coef_full.copy()

    cov_bayes, cov_bayes_space = _transform_covariance_to_prediction_space(
        fit_result.cov_bayes,
        P,
        source_space=getattr(fit_result, "cov_bayes_space", None),
    )
    cov_freq, cov_freq_space = _transform_covariance_to_prediction_space(
        fit_result.cov_freq,
        P,
        source_space=getattr(fit_result, "cov_freq_space", None),
    )
    cov_unconditional, cov_unconditional_space = (
        _transform_covariance_to_prediction_space(
            fit_result.cov_unconditional,
            P,
            source_space=getattr(fit_result, "cov_unconditional_space", None),
        )
    )

    beta = np.asarray(coef_full[_coef_column_offset(model) :], dtype=np.float64)
    return replace(
        fit_result,
        coef_full=coef_full,
        intercept=float(coef_full[0]) if _fit_intercept(model) else 0.0,
        beta=beta,
        cov_bayes=cov_bayes,
        cov_freq=cov_freq,
        cov_unconditional=cov_unconditional,
        coef_space=PREDICTION_PARAMETER_SPACE,
        cov_bayes_space=cov_bayes_space,
        cov_freq_space=cov_freq_space,
        cov_unconditional_space=cov_unconditional_space,
        penalty_quadratic=fit_result.penalty_quadratic,
    )


__all__ = [
    "FIT_PARAMETER_SPACE",
    "PREDICTION_PARAMETER_SPACE",
    "export_fit_result_to_prediction_space",
    "normalize_parameter_space",
    "prediction_parameterization_map",
]
