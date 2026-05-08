"""Internal data coercion and formula prediction helpers for GAM models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coerce_X(model, X, *, allow_missing_non_numeric: bool = False):
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_np = dataframe_to_feature_matrix(
            X,
            allow_missing_non_numeric=allow_missing_non_numeric,
        )
    else:
        X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")

        feature_names = [f"x{i}" for i in range(X_np.shape[1])]
        if X_np.dtype != object:
            X_num = np.asarray(X_np, dtype=np.float64)
            if not np.isfinite(X_num).all():
                raise ValueError("X contains NaN or Inf")
            X_np = X_num

    return X_np, feature_names


def coerce_feature_matrix(model, X):
    X_np, _ = coerce_X(model, X)
    if np.asarray(X_np).ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    return X_np


def coerce_optional_offset(offset, n_rows, *, name: str = "offset"):
    if offset is None:
        return None
    out = np.asarray(offset, dtype=np.float64).ravel()
    if out.shape != (int(n_rows),):
        raise ValueError(f"{name} must have shape ({int(n_rows)},), got {out.shape}.")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def _is_offset_sequence(offset):
    return isinstance(offset, (list, tuple))


def copy_offset(offset):
    if offset is None:
        return None
    if _is_offset_sequence(offset):
        return [
            None if off is None else np.asarray(off, dtype=np.float64).copy()
            for off in offset
        ]
    return np.asarray(offset, dtype=np.float64).copy()


def combine_offsets(*offsets):
    active = [o for o in offsets if o is not None]
    if not active:
        return None
    if any(_is_offset_sequence(offset) for offset in active):
        n_pred = max(
            len(offset) if _is_offset_sequence(offset) else 1 for offset in active
        )
        return [
            combine_offsets(
                *[
                    (
                        offset[i]
                        if _is_offset_sequence(offset) and i < len(offset)
                        else (
                            offset
                            if not _is_offset_sequence(offset) and i == 0
                            else None
                        )
                    )
                    for offset in active
                ]
            )
            for i in range(n_pred)
        ]
    arrs = [np.asarray(o, dtype=np.float64) for o in active]
    out = np.zeros_like(arrs[0], dtype=np.float64)
    for arr in arrs:
        if arr.shape != out.shape:
            raise ValueError(
                f"Offset arrays must all have the same shape, got {out.shape} and {arr.shape}."
            )
        out = out + arr
    return out


def knots_for_feature(model, feature_name, *, knots=None):
    knots = model.knots if knots is None else knots
    if knots is None:
        return None
    if isinstance(knots, dict):
        return knots.get(str(feature_name), None)
    return knots


def knots_for_features(model, feature_names, *, knots=None):
    knots = model.knots if knots is None else knots
    if knots is None:
        return None
    if isinstance(knots, dict):
        vals = [knots.get(str(f), None) for f in feature_names]
        return None if all(v is None for v in vals) else vals
    return knots


def dataframe_to_feature_matrix(
    X_df: pd.DataFrame, *, allow_missing_non_numeric: bool = False
):
    non_numeric = [
        c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
    ]

    if len(non_numeric) == 0:
        X_np = X_df.to_numpy(dtype=np.float64)
        if not np.isfinite(X_np).all():
            raise ValueError("X contains NaN or Inf")
        return X_np

    for c in X_df.columns:
        s = X_df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = np.asarray(s, dtype=np.float64)
            if not np.isfinite(vals).all():
                raise ValueError(f"Numeric column {c!r} contains NaN or Inf.")
        elif s.isna().any() and not bool(allow_missing_non_numeric):
            raise ValueError(
                f"Non-numeric column {c!r} contains missing values, "
                "which are not currently supported in fitting."
            )

    return X_df.to_numpy(dtype=object)


def coerce_formula_predict_inputs(model, X):
    from .specs.build import apply_formula_preprocess_to_new_data

    if not model.formula_mode_:
        X_np, feature_names = coerce_X(model, X)
        return X_np, feature_names, None

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Prediction for formula-based GAMs currently requires a pandas DataFrame."
        )

    X_work = apply_formula_preprocess_to_new_data(X, model.formula_preprocess_state_)
    feature_columns = getattr(model, "formula_feature_columns_", None)
    if feature_columns is None:
        feature_columns = model.formula_used_columns_
    missing = [c for c in feature_columns if c not in X_work.columns]
    if missing:
        raise KeyError(f"Prediction data is missing formula columns: {missing}")

    X_df = X_work[feature_columns]
    X_np, _ = coerce_X(model, X_df, allow_missing_non_numeric=True)

    offset_names = getattr(model, "formula_offset_names_", None)
    offset = None
    if offset_names is not None:
        if len(offset_names) == 1:
            offset_name = offset_names[0]
            if offset_name is not None:
                if offset_name not in X_work.columns:
                    raise KeyError(
                        "Prediction data is missing formula offset column: "
                        f"{offset_name!r}"
                    )
                if not pd.api.types.is_numeric_dtype(X_work[offset_name]):
                    raise NotImplementedError(
                        "Current formula-based prediction supports numeric offsets only. "
                        f"Offset column {offset_name!r} is non-numeric."
                    )
                offset = X_work[offset_name].to_numpy(dtype=np.float64)
        else:
            offset_list = []
            has_offset = False
            for offset_name in offset_names:
                if offset_name is None:
                    offset_list.append(None)
                    continue
                if offset_name not in X_work.columns:
                    raise KeyError(
                        "Prediction data is missing formula offset column: "
                        f"{offset_name!r}"
                    )
                if not pd.api.types.is_numeric_dtype(X_work[offset_name]):
                    raise NotImplementedError(
                        "Current formula-based prediction supports numeric offsets only. "
                        f"Offset column {offset_name!r} is non-numeric."
                    )
                offset_list.append(X_work[offset_name].to_numpy(dtype=np.float64))
                has_offset = True
            if has_offset:
                offset = offset_list

    return X_np, list(X_df.columns), offset
