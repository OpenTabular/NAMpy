"""Pure table adaptation shared by estimators and diagnostics."""

import numpy as np
import pandas as pd


def frame_for_estimator(estimator, X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional DataFrame or array.")
    schema = getattr(estimator, "schema_", None)
    names = getattr(schema, "feature_names", None)
    if names is None:
        names = getattr(estimator, "feature_names_in_", None)
    if names is None or len(names) != values.shape[1]:
        names = tuple(f"x{index}" for index in range(values.shape[1]))
    return pd.DataFrame(values, columns=names)


def _as_2d_array(X) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional table with shape "
            f"(n_samples, n_features); received shape {array.shape}."
        )
    return array


def feature_frame(X, feature_names):
    """Retain named rows or label a validated positional feature matrix."""
    return (
        X
        if isinstance(X, pd.DataFrame)
        else pd.DataFrame(_as_2d_array(X), columns=list(feature_names))
    )
