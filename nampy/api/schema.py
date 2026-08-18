"""Feature-schema contracts shared by NAMpy model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _feature_metadata(
    X: Any,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Return feature names, dtype names, and whether ``X`` has named columns."""
    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X.ndim} dimensions.")
        n_features = int(X.shape[1])
        return (
            tuple(f"x{index}" for index in range(n_features)),
            tuple(str(X.dtype) for _ in range(n_features)),
            False,
        )

    columns = getattr(X, "columns", None)
    dtypes = getattr(X, "dtypes", None)
    ndim = getattr(X, "ndim", None)
    if columns is None or dtypes is None or ndim is None:
        raise TypeError("X must be a pandas DataFrame or a NumPy ndarray.")
    if ndim != 2:
        raise ValueError(f"Expected a 2D DataFrame, got {ndim} dimensions.")

    return (
        tuple(str(column) for column in columns),
        tuple(str(dtype) for dtype in dtypes),
        True,
    )


@dataclass(frozen=True)
class FeatureSchema:
    """Names and dtypes recorded for a two-dimensional feature matrix."""

    feature_names: tuple[str, ...]
    dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.dtypes):
            raise ValueError(
                "Feature names and dtypes must contain the same number of entries."
            )

    @property
    def n_features(self) -> int:
        """Number of features represented by the schema."""
        return len(self.feature_names)

    @classmethod
    def from_data(cls, X: Any) -> FeatureSchema:
        """Build a schema from a pandas DataFrame or two-dimensional ndarray."""
        feature_names, dtypes, _ = _feature_metadata(X)
        return cls(feature_names=feature_names, dtypes=dtypes)

    def validate(self, X: Any) -> None:
        """Reject feature-count changes and named-column changes."""
        feature_names, _, has_named_columns = _feature_metadata(X)
        observed_count = len(feature_names)
        if observed_count != self.n_features:
            raise ValueError(
                "Feature count mismatch: "
                f"expected {self.n_features}, got {observed_count}."
            )
        if has_named_columns and feature_names != self.feature_names:
            raise ValueError(
                "Feature name mismatch: "
                f"expected {self.feature_names}, got {feature_names}."
            )
