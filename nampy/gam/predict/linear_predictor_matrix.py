"""
Linear predictor matrix (lpmatrix) construction for prediction.

The linear predictor matrix ``Xp`` satisfies ``eta = Xp @ coef_full``.
It is built by running new data through the compiled predictor's
``build_new_matrix`` method and prepending the intercept column if needed.

:func:`build_lpmatrix` is the public entry point.
"""

import numpy as np

from .._model_state import (
    _coerce_feature_matrix,
    _compiled_model,
    _design_matrix,
    _fit_intercept,
    _require_design,
    _require_fitted,
)
from .general import build_general_lpmatrix


def _build_prediction_matrices(model, X_new=None):
    _require_fitted(model)
    _require_design(model)
    compiled_model = _compiled_model(model)
    use_training_prediction_matrix = bool(
        compiled_model is not None
        and any(
            bool(getattr(tb, "metadata", {}).get("expose_raw_prediction_basis"))
            for tb in getattr(compiled_model, "compiled_terms", ())
        )
    )

    if X_new is None:
        if use_training_prediction_matrix:
            Z_new = np.asarray(compiled_model.build_new_matrix(model.X_), dtype=np.float64)
        else:
            Z_new = np.asarray(_design_matrix(model), dtype=np.float64)
    else:
        X_new = _coerce_feature_matrix(model, X_new, none_is_training=False)
        Z_new = compiled_model.build_new_matrix(X_new)

    if _fit_intercept(model):
        Xp = np.column_stack([np.ones(Z_new.shape[0], dtype=np.float64), Z_new])
    else:
        Xp = Z_new

    return Z_new, Xp


def build_lpmatrix(model, X_new=None):
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        return np.asarray(build_general_lpmatrix(model, X_new=X_new), dtype=np.float64)
    _, Xp = _build_prediction_matrices(model, X_new=X_new)
    return np.asarray(Xp, dtype=np.float64)


__all__ = ["build_lpmatrix", "_build_prediction_matrices"]
