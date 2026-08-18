"""Shared feature-effect plotting for the neural estimator wrappers.

The regressor, classifier, and LSS wrappers plot the same thing — per-feature
contribution curves and pairwise interaction heatmaps — differing only in how
the output columns are labelled. Wrappers supply those labels via their
``_plot_series_labels`` hook; everything else lives here once.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..neural.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)


def _as_2d_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def _reduce_targets_1d(y_true) -> np.ndarray:
    y = np.asarray(y_true)
    return y.mean(axis=1) if y.ndim > 1 else y


def plot_single_feature_effects(
    x_plot,
    predictions,
    y_true,
    ax,
    *,
    feature_name=None,
    num_bins=30,
    series_labels=None,
):
    """Plot one feature's contribution curve(s) with target scatter behind."""
    predictions = _as_2d_numpy(predictions)
    n_series = predictions.shape[1]

    y_1d = _reduce_targets_1d(y_true)
    y_true_centered = y_1d - np.mean(y_1d)
    y_range = (y_true_centered.min() - 1, y_true_centered.max() + 1)

    plot_density_shading(ax, x_plot, y_range, num_bins)

    if series_labels is None and n_series == 1:
        ax.plot(x_plot, predictions[:, 0], color="black", label="Shape Function")
    else:
        for i in range(n_series):
            label = (
                series_labels[i]
                if series_labels is not None and i < len(series_labels)
                else f"Output {i + 1}"
            )
            ax.plot(x_plot, predictions[:, i], label=label)

    ax.scatter(
        x_plot, y_true_centered, color="gray", alpha=0.3, s=2, label="True Values"
    )

    ax.set_title(
        f"Shape Function: {feature_name}" if feature_name else "Shape Function"
    )
    ax.set_xlabel(feature_name or "Feature")
    ax.set_ylabel("Contribution")
    ax.legend()


def plot_interaction_effects(
    interaction_name,
    interaction_preds,
    *,
    X_train_scaled=None,
    num_bins=30,
    series_labels=None,
):
    """Plot one pairwise interaction as binned heatmap(s)."""
    feature1, feature2 = interaction_name.split(":")

    interaction_preds = _as_2d_numpy(interaction_preds)
    n_series = interaction_preds.shape[1]

    x1_vals = (
        X_train_scaled[feature1].values
        if X_train_scaled is not None
        else np.arange(len(interaction_preds))
    )
    x2_vals = (
        X_train_scaled[feature2].values
        if X_train_scaled is not None
        else np.arange(len(interaction_preds))
    )

    fig, axes = create_subplot_grid(n_series)

    x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
    x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
    x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
    x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

    for out_idx, ax in enumerate(axes[:n_series]):
        contribs = interaction_preds[:, out_idx]

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
        plt.colorbar(im, ax=ax, label="Contribution")
        title = f"Interaction: {feature1} × {feature2}"
        if n_series > 1:
            label = (
                series_labels[out_idx]
                if series_labels is not None and out_idx < len(series_labels)
                else f"Output {out_idx + 1}"
            )
            title += f" ({label})"
        ax.set_title(title)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)

    for ax in axes[n_series:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_feature_effects(
    estimator, X, y_true, feature_name=None, plot_interactions=False
):
    """Shared body of the wrappers' ``plot`` method."""
    X_prepared, num_feature_names = prepare_plot_data(
        X,
        estimator.data_module.num_feature_info,
        estimator.data_module.cat_feature_info,
    )

    if feature_name is not None and feature_name not in num_feature_names:
        raise ValueError(
            f"Feature '{feature_name}' not found. Available: {num_feature_names}"
        )

    features_to_plot = [feature_name] if feature_name else num_feature_names
    predictions = estimator._predict(X_prepared)

    features_to_plot = [f for f in features_to_plot if f in predictions]
    if not features_to_plot:
        raise ValueError("No features found with predictions to plot.")

    fig, axes = create_subplot_grid(len(features_to_plot))

    for ax, fname in zip(axes, features_to_plot, strict=False):
        contribs = _as_2d_numpy(predictions[fname])
        series_labels = estimator._plot_series_labels(contribs.shape[1])
        plot_single_feature_effects(
            X_prepared[fname].values,
            contribs,
            y_true,
            ax,
            feature_name=fname,
            series_labels=series_labels,
        )

    for ax in axes[len(features_to_plot) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    if plot_interactions:
        for interaction_name in predictions.keys():
            if ":" in interaction_name:
                contribs = _as_2d_numpy(predictions[interaction_name])
                plot_interaction_effects(
                    interaction_name,
                    contribs,
                    X_train_scaled=X_prepared,
                    series_labels=estimator._plot_series_labels(contribs.shape[1]),
                )
