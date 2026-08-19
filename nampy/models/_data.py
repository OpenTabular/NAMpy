from __future__ import annotations

import numpy as np
import pandas as pd

from .._contracts import FeatureSchema


def _as_2d_array(X) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional table with shape "
            f"(n_samples, n_features); received shape {array.shape}."
        )
    return array


def _stringify_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column) for column in result.columns]
    if result.columns.has_duplicates:
        raise ValueError("Feature names must be unique after conversion to strings.")
    return result


def prepare_fit_features(estimator, X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = _stringify_columns(X)
    else:
        array = _as_2d_array(X)
        frame = pd.DataFrame(
            array,
            columns=[f"feature_{index}" for index in range(array.shape[1])],
        )

    estimator.n_features_in_ = int(frame.shape[1])
    estimator.feature_names_in_ = np.asarray(frame.columns, dtype=object)
    estimator._input_feature_names = list(frame.columns)
    estimator.schema_ = FeatureSchema.from_data(frame)
    return frame


def prepare_predict_features(estimator, X) -> pd.DataFrame:
    expected = getattr(estimator, "_input_feature_names", None)
    if expected is None:
        raise ValueError("The estimator has not been fitted yet.")

    if isinstance(X, pd.DataFrame):
        frame = _stringify_columns(X)
        estimator.schema_.validate(frame)
        return frame

    array = _as_2d_array(X)
    if array.shape[1] != len(expected):
        raise ValueError(
            f"X has {array.shape[1]} features, but the estimator was fitted with "
            f"{len(expected)} features."
        )
    estimator.schema_.validate(array)
    return pd.DataFrame(array, columns=expected)
