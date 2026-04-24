from __future__ import annotations

import numpy as np
import pandas as pd


def as_object_1d(x):
    arr = np.asarray(x, dtype=object).ravel()
    if arr.ndim != 1:
        raise ValueError("Expected a 1D vector.")
    return arr


def try_numeric_1d(x):
    try:
        vals = np.asarray(x, dtype=np.float64).ravel()
    except Exception:
        return None
    if vals.ndim != 1:
        return None
    return vals


def all_bool_like(arr):
    arr = as_object_1d(arr)
    return all(
        (v is None) or pd.isna(v) or isinstance(v, (bool, np.bool_)) for v in arr
    )


def is_factor_like_vector(x):
    """
    Factor-like means categorical / non-numeric input.
    Integer-coded columns remain numeric unless explicitly stored as
    categorical/object/string/bool.
    """
    x_obj = as_object_1d(x)

    if all_bool_like(x_obj):
        return True

    x_num = try_numeric_1d(x_obj)
    if x_num is not None and np.isfinite(x_num).all():
        return False
    return True


def factor_levels_from_metadata(metadata, feature_name):
    meta = dict(metadata or {}).get("factor_levels_by_feature", {})
    item = meta.get(str(feature_name), None)
    if item is None:
        return None
    if isinstance(item, dict):
        levels = item.get("levels", None)
    else:
        levels = item
    if levels is None:
        return None
    return list(levels)


def stable_unique_levels(x, levels=None):
    """
    Prefer sorted unique levels when possible, otherwise preserve first appearance.
    """
    if levels is not None:
        return list(levels)
    x = as_object_1d(x)
    try:
        return list(np.unique(x))
    except Exception:
        seen = []
        for v in x:
            matched = False
            for s in seen:
                try:
                    if v == s:
                        matched = True
                        break
                except Exception:
                    pass
            if not matched:
                seen.append(v)
        return seen


def factor_indicator_matrix(x, levels):
    """
    Full dummy coding over the supplied level set.
    Missing or unseen levels map to all-zero rows.
    """
    x = as_object_1d(x)
    levels = list(levels)
    idx = {lev: j for j, lev in enumerate(levels)}
    B = np.zeros((x.size, len(levels)), dtype=np.float64)

    for i, v in enumerate(x):
        if pd.isna(v):
            continue
        j = idx.get(v, None)
        if j is not None:
            B[i, j] = 1.0
    return B
