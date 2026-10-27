"""
Linear predictor matrix (lpmatrix) construction for prediction.

The linear predictor matrix ``Xp`` satisfies ``eta = Xp @ coef_full``.
It is built by running new data through the compiled predictor's
``build_new_matrix`` method and prepending the intercept column if needed.

:func:`build_lpmatrix` is the public entry point.
"""

import numpy as np

from ..model_state import (
    _coerce_feature_matrix,
    _compiled_model,
    _fit_intercept,
    _require_design,
    _require_fitted,
)
from .general import build_general_lpmatrix


def _build_prediction_matrices(
    model, X_new=None, *, skip_term_ids=(), allow_missing_numeric=False
):
    _require_fitted(model)
    _require_design(model)
    compiled_model = _compiled_model(model)
    if X_new is not None and not allow_missing_numeric:
        X_new = _coerce_feature_matrix(model, X_new, none_is_training=False)
    return compiled_prediction_matrices(
        compiled_model,
        X_new,
        fit_intercept=_fit_intercept(model),
        training_features=model.X_,
        skip_term_ids=skip_term_ids,
    )


def compiled_prediction_matrices(
    compiled_model,
    X_new,
    *,
    fit_intercept,
    training_features=None,
    skip_term_ids=(),
    fit_coordinates=False,
):
    """Build basis matrices independently of fitted coefficients/covariance.

    The GAM facade retains its public prediction coordinates. Loss-based fits
    request fit coordinates when the compiler exposes a wider prediction basis.
    """
    use_training_prediction_matrix = bool(
        compiled_model is not None
        and any(
            bool(getattr(tb, "metadata", {}).get("expose_raw_prediction_basis"))
            for tb in getattr(compiled_model, "compiled_terms", ())
        )
    )

    if X_new is None:
        if use_training_prediction_matrix:
            Z_new = np.asarray(
                compiled_model.build_new_matrix(training_features), dtype=np.float64
            )
        else:
            Z_new = np.asarray(compiled_model.design_matrix, dtype=np.float64)
    else:
        X_new = np.asarray(X_new)
        if X_new.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        Z_new = compiled_model.build_new_matrix(X_new, skip_term_ids=skip_term_ids)

    if fit_intercept:
        Xp = np.column_stack([np.ones(Z_new.shape[0], dtype=np.float64), Z_new])
    else:
        Xp = Z_new

    mapping = compiled_model.fit_to_prediction_parameterization_map
    if fit_coordinates and mapping is not None:
        Xp = Xp @ mapping

    return Z_new, Xp


def build_lpmatrix(model, X_new=None):
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        return np.asarray(build_general_lpmatrix(model, X_new=X_new), dtype=np.float64)
    _, Xp = _build_prediction_matrices(model, X_new=X_new)
    return np.asarray(Xp, dtype=np.float64)


__all__ = ["build_lpmatrix", "_build_prediction_matrices"]
