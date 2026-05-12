"""Generic interpretability helpers for fitted NAMpy estimators."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .plotting import create_subplot_grid


def _check_fitted(estimator):
    if (
        getattr(estimator, "model", None) is None
        or getattr(estimator, "data_module", None) is None
    ):
        raise ValueError("The model has not been fitted yet.")


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim == 1:
        value = value[:, np.newaxis]
    return value


def _broadcast_intercept(value, n_rows):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.reshape(1, -1)
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)

    if value.shape[0] == 1:
        value = np.repeat(value, n_rows, axis=0)
    elif value.shape[0] != n_rows:
        raise ValueError(
            "Intercept must be output-shaped or sample-aligned; "
            f"got first dimension {value.shape[0]} for {n_rows} samples."
        )
    return value


def _prepare_frame(estimator, X):
    feature_names = getattr(estimator, "feature_names_in_", None)
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    frame.columns = [str(column) for column in frame.columns]

    if feature_names is None:
        return frame

    feature_names = list(feature_names)
    if not isinstance(X, pd.DataFrame):
        if frame.shape[1] != len(feature_names):
            raise ValueError(
                "X has a different number of features than the fitted data: "
                f"got {frame.shape[1]}, expected {len(feature_names)}."
            )
        frame.columns = feature_names
        return frame

    missing = [name for name in feature_names if name not in frame.columns]
    extra = [name for name in frame.columns if name not in feature_names]
    if missing or extra:
        raise ValueError(
            "X feature names do not match the fitted data. "
            f"Missing: {missing}; extra: {extra}."
        )
    return frame.loc[:, feature_names]


def _normalize_terms(terms: Optional[Iterable[str]], available_terms):
    if terms is None:
        return list(available_terms)
    terms = [terms] if isinstance(terms, str) else list(terms)
    missing = [term for term in terms if term not in available_terms]
    if missing:
        raise ValueError(
            f"Unknown terms {missing}. Available terms: {available_terms}."
        )
    return terms


def _terms_to_frame(terms):
    columns = {}
    for term, values in terms.items():
        values = _as_numpy(values)
        if values.shape[1] == 1:
            columns[term] = values[:, 0]
        else:
            for idx in range(values.shape[1]):
                columns[f"{term}__output_{idx}"] = values[:, idx]
    return pd.DataFrame(columns)


def predict_terms(
    estimator,
    X,
    *,
    include_prediction: bool = False,
    include_intercept: bool = False,
    as_frame: bool = False,
):
    """
    Return per-term model contributions for a fitted NAMpy estimator.

    Contributions are on the model's raw additive scale: regression outputs,
    classification logits, or LSS distribution-parameter logits.
    """
    _check_fitted(estimator)
    predictions = estimator._predict(X)

    terms = {}
    n_rows = _as_numpy(predictions["prediction"]).shape[0]
    if include_prediction:
        terms["prediction"] = _as_numpy(predictions["prediction"])
    terms.update(
        {key: _as_numpy(value) for key, value in predictions.get("terms", {}).items()}
    )
    if include_intercept and predictions.get("intercept") is not None:
        terms["intercept"] = _broadcast_intercept(predictions["intercept"], n_rows)

    if as_frame:
        return _terms_to_frame(terms)
    return terms


def term_contributions(estimator, X, **kwargs):
    """Alias for :func:`predict_terms`."""
    return predict_terms(estimator, X, **kwargs)


def feature_importance(
    estimator,
    X,
    *,
    method: str = "variance",
    normalize: bool = True,
    include_interactions: bool = True,
):
    """
    Compute simple global term importance from per-sample contributions.

    Parameters
    ----------
    method : {"variance", "range", "mean_abs", "max_abs"}
        Aggregation used across samples for each output dimension.
    normalize : bool, default=True
        If True, divide importances by their total.
    include_interactions : bool, default=True
        Whether interaction terms such as ``"x1:x2"`` are included.
    """
    terms = predict_terms(estimator, X)
    rows = []

    for term, values in terms.items():
        if (not include_interactions) and ":" in term:
            continue

        values = _as_numpy(values)
        if method == "variance":
            per_output = np.var(values, axis=0)
        elif method == "range":
            per_output = np.ptp(values, axis=0)
        elif method == "mean_abs":
            per_output = np.mean(np.abs(values), axis=0)
        elif method == "max_abs":
            per_output = np.max(np.abs(values), axis=0)
        else:
            raise ValueError(
                "method must be one of 'variance', 'range', 'mean_abs', or 'max_abs'."
            )

        rows.append(
            {
                "term": term,
                "importance": float(np.mean(per_output)),
                "n_outputs": int(values.shape[1]),
            }
        )

    result = pd.DataFrame(rows, columns=["term", "importance", "n_outputs"])
    if result.empty:
        return result

    if normalize:
        total = result["importance"].sum()
        if total > 0:
            result["importance"] = result["importance"] / total

    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_terms(
    estimator,
    X,
    *,
    terms: Optional[Iterable[str]] = None,
    center: bool = True,
    max_cols: int = 4,
):
    """
    Plot per-term contributions for a fitted NAMpy estimator.

    Terms matching original feature columns are plotted against that feature;
    other terms are plotted against sample index.
    """
    terms_dict = predict_terms(estimator, X)
    available_terms = list(terms_dict)
    selected_terms = _normalize_terms(terms, available_terms)
    if not selected_terms:
        raise ValueError("No terms are available to plot.")
    X_frame = _prepare_frame(estimator, X)

    fig, axes = create_subplot_grid(len(selected_terms), max_cols=max_cols)
    for ax, term in zip(axes[: len(selected_terms)], selected_terms, strict=True):
        values = _as_numpy(terms_dict[term])
        if center:
            values = values - values.mean(axis=0, keepdims=True)

        if term in X_frame.columns:
            x_values = X_frame[term].to_numpy()
            order = np.argsort(x_values)
            x_plot = x_values[order]
            y_plot = values[order]
            ax.set_xlabel(term)
        else:
            x_plot = np.arange(values.shape[0])
            y_plot = values
            ax.set_xlabel("sample")

        for output_idx in range(y_plot.shape[1]):
            label = "contribution" if y_plot.shape[1] == 1 else f"output {output_idx}"
            ax.plot(x_plot, y_plot[:, output_idx], label=label)

        ax.set_title(term)
        ax.set_ylabel("contribution")
        if y_plot.shape[1] > 1:
            ax.legend()

    for ax in axes[len(selected_terms) :]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_interactions(
    estimator,
    X,
    *,
    terms: Optional[Iterable[str]] = None,
    num_bins: int = 30,
    max_cols: int = 4,
):
    """
    Plot pairwise interaction contributions as binned heatmaps.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")

    terms_dict = predict_terms(estimator, X)
    interaction_terms = [term for term in terms_dict if ":" in term]
    selected_terms = _normalize_terms(terms, interaction_terms)
    X_frame = _prepare_frame(estimator, X)

    plot_specs = []
    for term in selected_terms:
        feature_names = term.split(":")
        if len(feature_names) != 2:
            continue
        if any(feature not in X_frame.columns for feature in feature_names):
            continue
        values = _as_numpy(terms_dict[term])
        for output_idx in range(values.shape[1]):
            plot_specs.append((term, feature_names, output_idx, values[:, output_idx]))

    if not plot_specs:
        raise ValueError("No pairwise interaction terms can be plotted for this X.")

    fig, axes = create_subplot_grid(len(plot_specs), max_cols=max_cols)
    for ax, (term, feature_names, output_idx, contribs) in zip(
        axes[: len(plot_specs)], plot_specs, strict=True
    ):
        feature1, feature2 = feature_names
        x1_vals = X_frame[feature1].to_numpy()
        x2_vals = X_frame[feature2].to_numpy()

        x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
        x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
        x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
        x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

        grid_sum = np.zeros((num_bins - 1, num_bins - 1))
        grid_count = np.zeros((num_bins - 1, num_bins - 1), dtype=int)
        np.add.at(grid_sum, (x1_bin_idx, x2_bin_idx), contribs)
        np.add.at(grid_count, (x1_bin_idx, x2_bin_idx), 1)
        grid = np.where(grid_count > 0, grid_sum / np.maximum(grid_count, 1), np.nan)

        im = ax.imshow(
            grid.T,
            origin="lower",
            aspect="auto",
            extent=[x1_bins[0], x1_bins[-1], x2_bins[0], x2_bins[-1]],
            cmap="RdBu_r",
        )
        plt.colorbar(im, ax=ax, label="contribution")
        title = term if len(plot_specs) == 1 else f"{term} output {output_idx}"
        ax.set_title(title)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)

    for ax in axes[len(plot_specs) :]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes
