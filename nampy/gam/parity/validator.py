"""Single parity-validation entry point.

Reference target: mgcv 1.9-1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..results.snapshots import (
    _coerce_snapshot_arrays,
    build_parity_snapshot,
    load_parity_snapshot,
)
from .compare import compare_parity_snapshots


def _coerce_reference_snapshot(reference_fit: Any) -> dict[str, Any]:
    if isinstance(reference_fit, (str, Path)):
        snapshot: dict[str, Any] = load_parity_snapshot(reference_fit)
        return snapshot
    if isinstance(reference_fit, dict):
        return reference_fit
    raise TypeError(
        "reference_fit must be parity snapshot dict or path to saved parity snapshot."
    )


def _default_compare_X(model):
    if not bool(getattr(model, "formula_mode_", False)):
        return None

    used_columns = getattr(model, "formula_used_columns_", None)
    X_train = getattr(model, "X_", None)
    if used_columns is None or X_train is None:
        return None

    X_df = pd.DataFrame(X_train, columns=list(used_columns))
    offset_names = getattr(model, "formula_offset_names_", None)
    offset_default = getattr(model, "offset_predict_default_", None)
    if offset_names is not None and offset_default is not None:
        offset_values = (
            list(offset_default)
            if isinstance(offset_default, (list, tuple))
            else [offset_default]
        )
        for i, offset_name in enumerate(offset_names):
            if offset_name is None or i >= len(offset_values):
                continue
            offset_value = offset_values[i]
            if offset_value is None:
                continue
            X_df[offset_name] = offset_value
    return X_df


def compare(model, reference_fit, rtol=1e-6, atol=1e-8):
    """Compare fitted model against mgcv parity snapshot."""
    actual = _coerce_snapshot_arrays(
        build_parity_snapshot(model, X=_default_compare_X(model))
    )
    expected = _coerce_snapshot_arrays(_coerce_reference_snapshot(reference_fit))
    return compare_parity_snapshots(actual, expected, atol=atol, rtol=rtol)


__all__ = ["compare"]
