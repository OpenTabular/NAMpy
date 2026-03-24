"""
Linear predictor matrix (lpmatrix) construction for prediction.

The linear predictor matrix ``Xp`` satisfies ``eta = Xp @ coef_full``.
It is built by running new data through the compiled predictor's
``build_new_matrix`` method and prepending the intercept column if needed.

:func:`build_lpmatrix` is the public entry point.
"""

import numpy as np


def _coerce_prediction_matrix(model, X):
    if hasattr(model, "_coerce_feature_matrix"):
        return model._coerce_feature_matrix(X)

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    return X


def _build_prediction_matrices(model, X_new=None):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    if X_new is None:
        Z_new = model.Z
    else:
        X_new = _coerce_prediction_matrix(model, X_new)
        Z_new = model.design_.build_new_matrix(X_new)

    if model.fit_intercept:
        Xp = np.column_stack([np.ones(Z_new.shape[0], dtype=np.float64), Z_new])
    else:
        Xp = Z_new

    return Z_new, Xp


def build_lpmatrix(model, X_new=None):
    _, Xp = _build_prediction_matrices(model, X_new=X_new)
    return np.asarray(Xp, dtype=np.float64)


__all__ = ["build_lpmatrix", "_build_prediction_matrices"]
