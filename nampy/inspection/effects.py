"""Backend-neutral numerical preparation for additive interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _as_curve_matrix(values, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if array.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows, got {array.shape[0]}.")
    return array


@dataclass(frozen=True)
class DistributionalEffects:
    """Plot-ready point effects and conditional standard deviations.

    Each effect array may be ``(n_samples,)`` for one fit or
    ``(n_samples, n_fits)`` for repeated fits.  Keeping this data contract
    independent of an estimator lets archived results be rendered without
    loading a model.
    """

    feature_values: Mapping[str, np.ndarray]
    point_effects: Mapping[str, np.ndarray]
    conditional_sd: Mapping[str, np.ndarray]
    point_summary: str = "family_point"

    def __post_init__(self) -> None:
        if not isinstance(self.point_summary, str) or not self.point_summary:
            raise ValueError(
                "point_summary must name the family-defined point functional."
            )
        keys = tuple(self.point_effects)
        if not keys:
            raise ValueError("At least one distributional effect is required.")
        if set(self.feature_values) != set(keys) or set(self.conditional_sd) != set(
            keys
        ):
            raise ValueError(
                "feature_values, point_effects, and conditional_sd must have "
                "identical term names."
            )

        feature_values = {}
        point_effects = {}
        conditional_sd = {}
        repeat_count = None
        for term in keys:
            x = np.asarray(self.feature_values[term], dtype=np.float64).reshape(-1)
            point = _as_curve_matrix(
                self.point_effects[term], n_rows=len(x), name=f"point_effects[{term!r}]"
            )
            spread = _as_curve_matrix(
                self.conditional_sd[term],
                n_rows=len(x),
                name=f"conditional_sd[{term!r}]",
            )
            if point.shape != spread.shape:
                raise ValueError(
                    f"Point effects and conditional SD for {term!r} must have "
                    "matching shapes."
                )
            if point.shape[1] < 1:
                raise ValueError("Every term must contain at least one fitted curve.")
            if repeat_count is None:
                repeat_count = point.shape[1]
            elif point.shape[1] != repeat_count:
                raise ValueError("Every term must contain the same number of fits.")
            feature_values[term] = x
            point_effects[term] = point
            conditional_sd[term] = spread

        object.__setattr__(self, "feature_values", feature_values)
        object.__setattr__(self, "point_effects", point_effects)
        object.__setattr__(self, "conditional_sd", conditional_sd)

    @property
    def spread_kind(self) -> str:
        return "conditional_standard_deviation"

    @property
    def variability_kind(self) -> str:
        return "between_fits"

    @property
    def n_fits(self) -> int:
        """Number of fitted curves represented by each term."""
        first = next(iter(self.point_effects.values()))
        return int(np.asarray(first).shape[1])


def combine_distributional_effects(
    effects: Sequence[DistributionalEffects],
) -> DistributionalEffects:
    """Combine compatible single- or repeated-fit effect data column-wise."""
    items = tuple(effects)
    if not items:
        raise ValueError("effects must contain at least one item.")
    terms = tuple(items[0].point_effects)
    for current in items[1:]:
        if current.point_summary != items[0].point_summary:
            raise ValueError("Combined effects must use the same point functional.")
        if tuple(current.point_effects) != terms:
            raise ValueError("All effect sets must have identical ordered terms.")
        for term in terms:
            if (
                current.feature_values[term].shape
                != items[0].feature_values[term].shape
            ):
                raise ValueError(
                    f"Feature values for {term!r} differ between effect sets."
                )
            if not np.allclose(
                current.feature_values[term],
                items[0].feature_values[term],
                equal_nan=True,
            ):
                raise ValueError(
                    f"Feature values for {term!r} differ between effect sets."
                )
    return DistributionalEffects(
        feature_values=items[0].feature_values,
        point_summary=items[0].point_summary,
        point_effects={
            term: np.column_stack([item.point_effects[term] for item in items])
            for term in terms
        },
        conditional_sd={
            term: np.column_stack([item.conditional_sd[term] for item in items])
            for term in terms
        },
    )


def _binned_effects(x, point, spread, *, bins: int):
    table = pd.DataFrame({"x": x})
    try:
        table["bin"] = pd.qcut(table["x"], q=bins, duplicates="drop")
    except ValueError:
        table["bin"] = table["x"]
    point_columns = []
    spread_columns = []
    for index in range(point.shape[1]):
        point_name = f"point_{index}"
        spread_name = f"spread_{index}"
        table[point_name] = point[:, index]
        table[spread_name] = spread[:, index]
        point_columns.append(point_name)
        spread_columns.append(spread_name)
    grouped = table.groupby("bin", observed=True, dropna=True).mean(numeric_only=True)
    if grouped.empty:
        raise ValueError("A feature has no finite values to plot.")
    return (
        grouped["x"].to_numpy(dtype=np.float64),
        grouped[point_columns].to_numpy(dtype=np.float64),
        grouped[spread_columns].to_numpy(dtype=np.float64),
    )


def _frame_for_estimator(estimator, X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional DataFrame or array.")
    schema = getattr(estimator, "schema_", None)
    names = getattr(schema, "feature_names", None)
    if names is None or len(names) != values.shape[1]:
        names = tuple(f"x{index}" for index in range(values.shape[1]))
    return pd.DataFrame(values, columns=names)


def prepare_distributional_effects(
    estimator,
    X,
    *,
    reference_X=None,
    reference_weight=None,
    response_scale: float = 1.0,
    feature_names: Sequence[str] | None = None,
) -> DistributionalEffects:
    """Compute point effects and conditional SDs through a family contract.

    For each one-dimensional additive term, all other centered terms are held
    at their reference values.  The plotted point effect is the change in the
    family's ``predict_point`` relative to that reference; conditional spread
    comes from the family's ``predict_spread`` implementation.  Consequently,
    the plotting code does not inspect distribution parameter names or contain
    family-specific formulas.
    """
    scale = float(response_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("response_scale must be finite and positive.")

    frame = _frame_for_estimator(estimator, X)
    from .contributions import center_additive_prediction

    components = center_additive_prediction(
        estimator.predict_components(X),
        reference=None
        if reference_X is None
        else estimator.predict_components(reference_X),
        sample_weight=reference_weight,
    )
    components.validate_additive_reconstruction(atol=1e-5)
    link = np.asarray(components.link, dtype=np.float64)
    if link.ndim == 1:
        link = link[:, None]
    intercept = np.asarray(components.intercept, dtype=np.float64)
    baseline_raw = np.broadcast_to(intercept, link.shape).copy()
    if components.offset is not None:
        baseline_raw += components.offset
    baseline_point, _ = estimator.summarize_predictors(baseline_raw)

    requested = None if feature_names is None else tuple(feature_names)
    if requested is not None:
        unknown = set(requested) - set(components.terms)
        if unknown:
            raise ValueError(f"Unknown feature terms: {sorted(unknown)}")

    feature_values = {}
    point_effects = {}
    conditional_sd = {}
    for term, values in components.terms.items():
        if requested is not None and term not in requested:
            continue
        metadata = components.term_metadata.get(term)
        if metadata is not None:
            if metadata.kind != "main" or len(metadata.features) != 1:
                continue
            feature = metadata.features[0]
        else:
            feature = term
        if feature not in frame or not pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        contribution = np.asarray(values, dtype=np.float64)
        if contribution.ndim == 1:
            contribution = contribution[:, None]
        try:
            contribution = np.broadcast_to(contribution, link.shape)
        except ValueError as error:
            raise ValueError(
                f"Term {term!r} has shape {contribution.shape}, which cannot be "
                f"aligned with link shape {link.shape}."
            ) from error
        point, spread = estimator.summarize_predictors(baseline_raw + contribution)
        feature_values[term] = frame[feature].to_numpy(dtype=np.float64)
        point_effects[term] = (point - baseline_point) * scale
        conditional_sd[term] = spread * scale

    if not point_effects:
        raise ValueError("No numeric one-dimensional feature effects are available.")
    return DistributionalEffects(
        feature_values,
        point_effects,
        conditional_sd,
        point_summary=getattr(estimator, "predictor_point_summary", "family_point"),
    )


__all__ = [
    "DistributionalEffects",
    "combine_distributional_effects",
    "prepare_distributional_effects",
]
