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
    """
    Low-level linear predictor matrix builder.

    Parameters
    ----------
    model : fitted core/model object
    X_new : array-like or None
        If None, returns the in-sample lpmatrix.

    Returns
    -------
    np.ndarray
    """
    _, Xp = _build_prediction_matrices(model, X_new=X_new)
    return np.asarray(Xp, dtype=np.float64)