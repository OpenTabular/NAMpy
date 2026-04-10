"""Shared model lifecycle checks and duck-typed GAM wrapper accessors."""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_fitted(obj: Any) -> None:
    if not getattr(obj, "_fitted", False):
        raise RuntimeError("Model is not fitted.")


def _require_design(model: Any) -> None:
    if getattr(model, "design_", None) is None:
        raise RuntimeError("Model has no compiled design; fit the model first.")


def _fit_intercept(obj: Any) -> bool:
    return bool(getattr(obj, "fit_intercept", False))


def _coef_column_offset(obj: Any) -> int:
    return 1 if _fit_intercept(obj) else 0


def _term_blocks_seq(obj: Any):
    blocks = getattr(obj, "term_blocks_", None)
    if not blocks:
        return ()
    return blocks


def _coerce_feature_matrix(model: Any, X, *, none_is_training: bool = False):
    """Coerce user features to a 2D array; optional training-matrix default."""
    if none_is_training and X is None:
        return model.X_
    if hasattr(model, "_coerce_feature_matrix"):
        return model._coerce_feature_matrix(X)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    return X
