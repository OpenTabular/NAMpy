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
    _fit_intercept,
    _require_design,
    _require_fitted,
)


def _build_prediction_matrices(model, X_new=None):
    _require_fitted(model)
    _require_design(model)

    if X_new is None:
        Z_new = model.Z
    else:
        X_new = _coerce_feature_matrix(model, X_new, none_is_training=False)
        Z_new = model.design_.build_new_matrix(X_new)

    if _fit_intercept(model):
        Xp = np.column_stack([np.ones(Z_new.shape[0], dtype=np.float64), Z_new])
    else:
        Xp = Z_new

    return Z_new, Xp


def build_lpmatrix(model, X_new=None):
    _, Xp = _build_prediction_matrices(model, X_new=X_new)
    return np.asarray(Xp, dtype=np.float64)


__all__ = ["build_lpmatrix", "_build_prediction_matrices"]
