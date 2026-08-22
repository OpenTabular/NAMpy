"""Backend-neutral additive-model explanation tables."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .contracts import AdditivePrediction


def _weighted_term_mean(values, sample_weight=None):
    array = np.asarray(values, dtype=float)
    if sample_weight is None:
        return np.mean(array, axis=0)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weights) != array.shape[0]:
        raise ValueError("sample_weight and contributions must have the same rows.")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and non-negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("sample_weight must sum to a positive value.")
    return np.average(array, axis=0, weights=weights)


def center_additive_prediction(
    prediction: AdditivePrediction,
    *,
    reference: AdditivePrediction | None = None,
    sample_weight=None,
) -> AdditivePrediction:
    """Center every term and move its reference mean into the intercept.

    The link and response predictions remain unchanged, so additive
    reconstruction is preserved exactly up to floating-point arithmetic.
    """
    source = prediction if reference is None else reference
    if set(source.terms) != set(prediction.terms):
        raise ValueError("Prediction and reference must contain the same terms.")

    centered_terms = {}
    intercept = np.asarray(prediction.intercept, dtype=float)
    for name, values in prediction.terms.items():
        reference_values = np.asarray(source.terms[name])
        values_array = np.asarray(values)
        if reference_values.shape[1:] != values_array.shape[1:]:
            raise ValueError(f"Reference term {name!r} has an incompatible shape.")
        shift = _weighted_term_mean(reference_values, sample_weight)
        centered_terms[name] = values_array - shift
        intercept = intercept + shift

    if intercept.size == 1:
        centered_intercept = float(intercept.reshape(-1)[0])
    else:
        centered_intercept = intercept
    return AdditivePrediction(
        response=np.asarray(prediction.response),
        link=np.asarray(prediction.link),
        terms=centered_terms,
        intercept=centered_intercept,
        backend=prediction.backend,
        offset=prediction.offset,
    )


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional DataFrame or array.")
    return pd.DataFrame(
        values, columns=[f"x{index}" for index in range(values.shape[1])]
    )


def _term_features(term: str, columns) -> tuple[str, ...]:
    column_names = {str(column): str(column) for column in columns}
    if term in column_names:
        return (column_names[term],)

    colon_parts = tuple(part.strip() for part in term.split(":"))
    if len(colon_parts) > 1 and all(part in column_names for part in colon_parts):
        return tuple(column_names[part] for part in colon_parts)

    match = re.match(r"^(?:s|te|ti|t2)\((.*)\)$", term)
    if match:
        candidates = tuple(
            part.strip() for part in match.group(1).split(",") if part.strip()
        )
        resolved = tuple(
            column_names[candidate]
            for candidate in candidates
            if candidate in column_names
        )
        if resolved:
            return resolved
    return ()


def _term_type(term: str) -> str:
    if ":" in term or re.match(r"^(?:te|ti|t2)\(", term):
        return "interaction"
    return "main"


def _binned_values(values: pd.Series, max_bins: int):
    if max_bins < 1:
        raise ValueError("max_bins must be at least 1.")
    if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=False) > max_bins:
        bins = pd.qcut(values, q=max_bins, duplicates="drop")
        return bins.map(lambda interval: interval.mid if pd.notna(interval) else np.nan)
    return values


def term_importance_table(prediction: AdditivePrediction) -> pd.DataFrame:
    """Return mean absolute link-scale contribution by term and output."""
    rows = []
    for term, contribution in prediction.terms.items():
        values = np.asarray(contribution)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        term_type = _term_type(term)
        for output in range(values.shape[1]):
            rows.append(
                {
                    "term": term,
                    "term_type": term_type,
                    "output": output,
                    "importance": float(np.mean(np.abs(values[:, output]))),
                }
            )
    return pd.DataFrame(
        rows, columns=["term", "term_type", "output", "importance"]
    ).sort_values("importance", ascending=False, ignore_index=True)


def explain_additive_prediction(
    X,
    prediction: AdditivePrediction,
    *,
    max_bins: int = 64,
) -> pd.DataFrame:
    """Aggregate additive contributions into a model-independent term table.

    The returned schema follows the useful part of NODE-GAM's ``get_GAM_df``:
    term identity, term type, plotted feature values, mean contribution,
    observation count, and global mean-absolute contribution importance.
    Contributions and importance are always reported on the additive/link
    scale represented by :class:`AdditivePrediction`.
    """
    frame = _as_frame(X)
    if len(frame) != np.asarray(prediction.link).shape[0]:
        raise ValueError("X and prediction must contain the same number of rows.")
    importance = term_importance_table(prediction).set_index(
        ["term", "output"]
    )["importance"]
    tables = []

    for term, contribution in prediction.terms.items():
        values = np.asarray(contribution)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        features = _term_features(term, frame.columns)
        term_type = "interaction" if len(features) > 1 else _term_type(term)
        group_columns = []
        base = pd.DataFrame(index=frame.index)
        for index, feature in enumerate(features):
            value_name = "value" if index == 0 else f"value_{index + 1}"
            base[value_name] = _binned_values(frame[feature], max_bins)
            group_columns.append(value_name)

        # Unknown formula labels still retain a meaningful average row rather
        # than guessing which source columns define the term.
        if not group_columns:
            base["value"] = term
            group_columns = ["value"]

        for output in range(values.shape[1]):
            current = base.copy()
            current["contribution"] = values[:, output]
            grouped = (
                current.groupby(group_columns, dropna=False, observed=True)
                .agg(contribution=("contribution", "mean"), count=("contribution", "size"))
                .reset_index()
            )
            grouped.insert(0, "output", output)
            grouped.insert(0, "term_type", term_type)
            grouped.insert(0, "term", term)
            grouped["importance"] = importance.loc[(term, output)]
            tables.append(grouped)

    base_columns = [
        "term",
        "term_type",
        "output",
        "contribution",
        "count",
        "importance",
    ]
    if not tables:
        return pd.DataFrame(
            columns=[*base_columns[:3], "value", *base_columns[3:]]
        )
    result = pd.concat(tables, ignore_index=True, sort=False)
    value_columns = ["value"]
    value_columns.extend(
        sorted(
            (name for name in result if name.startswith("value_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
    )
    columns = [*base_columns[:3], *value_columns, *base_columns[3:]]
    return result.reindex(columns=columns)
