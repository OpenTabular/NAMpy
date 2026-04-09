"""Single parity-validation entry point.

Reference target: mgcv 1.9-1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .compare import compare_parity_snapshots
from .snapshots import (
    _coerce_snapshot_arrays,
    build_parity_snapshot,
    load_parity_snapshot,
)


def _coerce_reference_snapshot(mgcv_fit: Any) -> dict[str, Any]:
    if isinstance(mgcv_fit, (str, Path)):
        return load_parity_snapshot(mgcv_fit)
    if isinstance(mgcv_fit, dict):
        return mgcv_fit
    raise TypeError(
        "mgcv_fit must be parity snapshot dict or path to saved parity snapshot."
    )


def _default_compare_X(model):
    if not bool(getattr(model, "formula_mode_", False)):
        return None

    used_columns = getattr(model, "formula_used_columns_", None)
    X_train = getattr(model, "X_", None)
    if used_columns is None or X_train is None:
        return None

    X_df = pd.DataFrame(X_train, columns=list(used_columns))
    offset_name = getattr(model, "formula_offset_name_", None)
    offset_default = getattr(model, "offset_predict_default_", None)
    if offset_name is not None and offset_default is not None:
        X_df[offset_name] = offset_default
    return X_df


def compare(model, mgcv_fit, rtol=1e-6, atol=1e-8):
    """Compare fitted model against mgcv parity snapshot."""
    actual = _coerce_snapshot_arrays(
        build_parity_snapshot(model, X=_default_compare_X(model))
    )
    expected = _coerce_snapshot_arrays(_coerce_reference_snapshot(mgcv_fit))
    return compare_parity_snapshots(actual, expected, atol=atol, rtol=rtol)


__all__ = ["compare"]
