"""Small backend-neutral contracts shared by model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
                f"Feature dtype mismatch: expected {self.dtypes}, got {dtypes}."
            )


@dataclass(frozen=True)
class AdditiveTermMetadata:
    """Feature dependence and parameter identity, independent of display labels.

    ``kind`` distinguishes main effects, interactions, and global/contextual
    contributions. Reconstruction alone does not establish feature dependence.
    """

    features: tuple[str, ...] = ()
    parameter: str | None = None
    kind: Literal["main", "interaction", "global", "contextual"] = "main"


@dataclass(frozen=True)
class AdditivePrediction:
    """Additive predictors, independent of response and parameter dimensions.

    Link and term arrays are normalized to ``(n_samples, n_predictors)``.
    Intercepts may be predictor constants or rowwise values; offsets are
    rowwise predictor arrays. Names identify actual additive predictors.
    """

    link: np.ndarray
    terms: dict[str, np.ndarray]
    intercept: float | np.ndarray
    backend: Literal["gam", "neural"]
    offset: np.ndarray | None = None
    term_metadata: dict[str, AdditiveTermMetadata] = field(default_factory=dict)
    predictor_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in {"gam", "neural"}:
            raise ValueError("backend must be 'gam' or 'neural'.")
        link = np.asarray(self.link)
        if link.ndim == 1:
            link = link[:, None]
        if link.ndim != 2 or link.shape[1] < 1:
            raise ValueError("link must have shape (n_samples, n_predictors).")
        n, k = link.shape
        names = tuple(self.predictor_names) or tuple(f"eta{i + 1}" for i in range(k))
        if (
            len(names) != k
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != k
        ):
            raise ValueError("predictor_names must uniquely name every predictor.")

        def row_array(value, name):
            array = np.asarray(value)
            if array.shape == (n,):
                array = array[:, None]
            try:
                return np.broadcast_to(array, (n, k))
            except ValueError as error:
                raise ValueError(
                    f"{name} has an incompatible shape {array.shape}."
                ) from error

        terms = {}
        for name, value in self.terms.items():
            array = np.asarray(value)
            if array.shape == (n,):
                array = array[:, None]
            if array.shape != (n, k):
                raise ValueError(f"Term {name!r} must match link shape {(n, k)}.")
            terms[name] = array
        intercept = np.asarray(self.intercept)
        if intercept.ndim == 0:
            intercept = np.full(k, intercept.item())
        elif intercept.shape != (k,):
            intercept = row_array(intercept, "intercept")
        if set(self.term_metadata) - set(terms):
            raise ValueError("term_metadata contains unknown terms.")
        object.__setattr__(self, "link", link)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "predictor_names", names)
        if self.offset is not None:
            object.__setattr__(self, "offset", row_array(self.offset, "offset"))
        # Neural decompositions can contain float32 rounding from independent
        # term batches. Validate at their stored precision without modifying
        # any predictions or imposing float32 tolerance on statistical fits.
        arrays = [link, intercept, *terms.values()]
        if self.offset is not None:
            arrays.append(self.offset)
        precision = max(
            (
                np.finfo(array.dtype).eps
                for array in arrays
                if np.issubdtype(array.dtype, np.inexact)
            ),
            default=0.0,
        )
        self.validate_additive_reconstruction(
            rtol=max(1e-7, 16 * precision), atol=max(1e-9, 16 * precision)
        )

    def validate_additive_reconstruction(self, *, rtol=1e-7, atol=1e-9) -> None:
        """Validate reconstruction in predictor coordinates, preserving NA rows."""
        expected = np.broadcast_to(self.intercept, self.link.shape).copy()
        for term in self.terms.values():
            expected = expected + term
        if self.offset is not None:
            expected = expected + self.offset
        if not np.allclose(self.link, expected, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(
                "link is not reconstructed by intercept, terms, and offset."
            )


@dataclass(frozen=True)
class EnsembleAdditivePrediction:
    """Mean additive prediction and between-estimator standard deviations."""

    mean: AdditivePrediction
    link_std: np.ndarray
    term_std: dict[str, np.ndarray]
    intercept_std: float | np.ndarray
    n_estimators: int

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if np.asarray(self.link_std).shape != np.asarray(self.mean.link).shape:
            raise ValueError("link_std must match the mean link shape.")
        if set(self.term_std) != set(self.mean.terms):
            raise ValueError("term_std must contain every mean contribution term.")
        for name, values in self.term_std.items():
            if np.asarray(values).shape != np.asarray(self.mean.terms[name]).shape:
                raise ValueError(f"term_std for {name!r} has an incompatible shape.")


@dataclass(frozen=True)
class DistributionFitStatus:
    """Backend-owned eligibility evidence, separate from probability support.

    Statistical convergence and completed neural training are distinct states.
    Unknown, failed and nonconverged fits are not eligible for family ranking.
    """

    backend: str
    state: Literal["converged", "completed", "nonconverged", "failed", "unknown"]
    message: str = ""

    def __post_init__(self):
        if not self.backend or self.state not in {
            "converged",
            "completed",
            "nonconverged",
            "failed",
            "unknown",
        }:
            raise ValueError("Invalid distribution fit status.")

    @property
    def eligible(self) -> bool:
        return self.state in {"converged", "completed"}


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """Declared operation support and explicit conditions, without running it."""

    state: Literal["supported", "conditional", "unsupported"]
    reason: str = ""
    conditions: tuple[str, ...] = ()

    def __post_init__(self):
        if self.state not in {"supported", "conditional", "unsupported"}:
            raise ValueError("Unknown capability state.")
        conditions = tuple(self.conditions)
        if any(not isinstance(value, str) or not value for value in conditions):
            raise ValueError("Capability conditions must be nonempty strings.")
        object.__setattr__(self, "conditions", conditions)
