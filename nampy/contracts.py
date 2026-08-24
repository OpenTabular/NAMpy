"""Small backend-neutral contracts shared by model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
        """Reject feature-count, name/order, and dtype changes."""
        feature_names, dtypes, has_named_columns = _feature_metadata(X)
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
        if dtypes != self.dtypes:
            raise ValueError(
                "Feature dtype mismatch: "
                f"expected {self.dtypes}, got {dtypes}."
            )

@dataclass(frozen=True)
class AdditivePrediction:
    """Prediction outputs and additive contributions on the link scale."""

    response: np.ndarray
    link: np.ndarray
    terms: dict[str, np.ndarray]
    intercept: float | np.ndarray
    backend: Literal["gam", "neural"]
    offset: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"gam", "neural"}:
            raise ValueError("backend must be 'gam' or 'neural'.")

        response = np.asarray(self.response)
        link = np.asarray(self.link)
        if response.ndim not in {1, 2} or link.shape != response.shape:
            raise ValueError("response and link must have matching 1D or 2D shapes.")
        n_samples = response.shape[0]
        for name, value in self.terms.items():
            shape = np.asarray(value).shape
            if not shape or shape[0] != n_samples:
                raise ValueError(f"Term {name!r} has an incompatible sample shape.")
            if response.ndim == 2 and shape[1:] not in {(), (1,), response.shape[1:]}:
                raise ValueError(f"Term {name!r} has an incompatible output shape.")

        scalar_inputs: tuple[
            tuple[str, float | np.ndarray | None], ...
        ] = (("intercept", self.intercept), ("offset", self.offset))
        for name, scalar_value in scalar_inputs:
            if scalar_value is None:
                continue
            shape = np.asarray(scalar_value).shape
            compatible = {(), (n_samples,), response.shape}
            if response.ndim == 2:
                compatible.add((response.shape[1],))
            if shape not in compatible:
                raise ValueError(f"{name} has an incompatible shape {shape}.")

    def validate_additive_reconstruction(self, *, rtol=1e-7, atol=1e-9) -> None:
        """Validate the link-scale additive decomposition for this result."""
        link = np.asarray(self.link)

        def aligned(value):
            array = np.asarray(value, dtype=float)
            if link.ndim == 1 and array.shape == (link.shape[0], 1):
                return array[:, 0]
            if link.ndim == 2 and array.shape == (link.shape[0],):
                return array[:, None]
            return array

        expected = aligned(self.intercept)
        for term in self.terms.values():
            expected = expected + aligned(term)
        if self.offset is not None:
            expected = expected + aligned(self.offset)
        if not np.allclose(link, expected, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError("link is not reconstructed by intercept, terms, and offset.")


@dataclass(frozen=True)
class EnsembleAdditivePrediction:
    """Mean additive prediction and between-estimator standard deviations."""

    mean: AdditivePrediction
    response_std: np.ndarray
    link_std: np.ndarray
    term_std: dict[str, np.ndarray]
    intercept_std: float | np.ndarray
    n_estimators: int

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if np.asarray(self.response_std).shape != np.asarray(self.mean.response).shape:
            raise ValueError("response_std must match the mean response shape.")
        if np.asarray(self.link_std).shape != np.asarray(self.mean.link).shape:
            raise ValueError("link_std must match the mean link shape.")
        if set(self.term_std) != set(self.mean.terms):
            raise ValueError("term_std must contain every mean contribution term.")
        for name, values in self.term_std.items():
            if np.asarray(values).shape != np.asarray(self.mean.terms[name]).shape:
                raise ValueError(f"term_std for {name!r} has an incompatible shape.")
