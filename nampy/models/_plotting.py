"""Shared feature-effect plotting for the neural estimator wrappers.

The regressor, classifier, and LSS wrappers plot the same thing — per-feature
contribution curves, pairwise interaction heatmaps, and conditioned slices of
higher-order interactions — differing only in how the output columns are
labelled. Wrappers supply those labels via their ``_plot_series_labels`` hook;
everything else lives here once.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_subplot_grid(n_plots, max_cols=4, subplot_size=(5, 4)):
    """
    Create a figure with a grid of subplots.

    Parameters
    ----------
    n_plots : int
        Number of subplots to create.
    max_cols : int, optional
        Maximum number of columns, by default 4.
    subplot_size : tuple, optional
        (width, height) of each subplot in inches, by default (5, 4).

    Returns
    -------
    tuple
        (fig, axes) where axes is a flattened array of Axes objects.
    """
    ncols = min(n_plots, max_cols)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
        squeeze=False,  # Always return 2D array for consistent handling
    )
    return fig, axes.flatten()


def prepare_plot_data(X, num_feature_info, cat_feature_info):
    """
    Prepare and validate input data for plotting.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input data to prepare.
    num_feature_info : dict
        Dictionary of numerical feature information.
    cat_feature_info : dict
        Dictionary of categorical feature information.

    Returns
    -------
    tuple
        (X_prepared, num_feature_names) where X_prepared is the processed DataFrame.

    Raises
    ------
    ValueError
        If the input data has incorrect number of columns.
    """
    num_feature_names = list(num_feature_info.keys())
    cat_feature_names = list(cat_feature_info.keys())
    all_feature_names = num_feature_names + cat_feature_names

    X_df = pd.DataFrame(X)

    # Assign column names if needed
    if not all(col in all_feature_names for col in X_df.columns):
        if len(X_df.columns) != len(all_feature_names):
            raise ValueError(
                f"Input has {len(X_df.columns)} columns but model expects {len(all_feature_names)} features."
            )
        X_df.columns = all_feature_names

    # Sort numerical columns for smooth plotting
    for fname in num_feature_names:
        if fname in X_df.columns:
            X_df[fname] = X_df[fname].sort_values().values

    return X_df, num_feature_names


def plot_density_shading(ax, x_values, y_range, num_bins=30):
    """
    Add density-based background shading to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add shading to.
    x_values : np.ndarray
        The x values to compute density from.
    y_range : tuple
        (y_min, y_max) range for the shading bars.
    num_bins : int, optional
        Number of bins for density computation, by default 30.
    """
    counts, bin_edges = np.histogram(x_values, bins=num_bins)
    max_count = counts.max() if counts.size else 1
    norm_counts = counts / max_count

    for i in range(num_bins):
        ax.bar(
            bin_edges[i],
            y_range,
            width=bin_edges[i + 1] - bin_edges[i],
            color=plt.cm.Reds(norm_counts[i]),
            alpha=0.6,
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
    slice_bins=3,
    max_slices=9,
):
    """Plot pairwise heatmaps or observed slices of a higher-order term."""
    features = interaction_name.split(":")
    if len(features) < 2:
        raise ValueError("An interaction plot requires at least two features.")
    feature1, feature2 = features[:2]

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

    x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
    x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
    x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
    x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

    slice_masks = [("", np.ones(len(interaction_preds), dtype=bool))]
    if len(features) > 2:
        if X_train_scaled is None:
            raise ValueError(
                "Higher-order interaction slices require source feature values."
            )
        if slice_bins < 1 or max_slices < 1:
            raise ValueError("slice_bins and max_slices must be positive.")
        conditions = pd.DataFrame(index=np.arange(len(interaction_preds)))
        for feature in features[2:]:
            values = X_train_scaled[feature]
            if pd.api.types.is_numeric_dtype(values):
                conditions[feature] = pd.qcut(
                    values, q=slice_bins, duplicates="drop"
                ).astype(str)
            else:
                conditions[feature] = values.astype(str)
        grouped = conditions.groupby(list(conditions), dropna=False).groups
        ranked_groups = sorted(
            grouped.items(), key=lambda item: (-len(item[1]), str(item[0]))
        )[:max_slices]
        slice_masks = []
        for condition, row_indices in ranked_groups:
            values = condition if isinstance(condition, tuple) else (condition,)
            label = ", ".join(
                f"{name}={value}" for name, value in zip(features[2:], values, strict=True)
            )
            mask = np.zeros(len(interaction_preds), dtype=bool)
            mask[np.asarray(row_indices, dtype=int)] = True
            slice_masks.append((label, mask))

    fig, axes = create_subplot_grid(n_series * len(slice_masks))

    for slice_index, (slice_label, slice_mask) in enumerate(slice_masks):
        for out_idx in range(n_series):
            ax = axes[slice_index * n_series + out_idx]
            contribs = interaction_preds[slice_mask, out_idx]

            grid_sum = np.zeros((num_bins - 1, num_bins - 1))
            grid_count = np.zeros((num_bins - 1, num_bins - 1), dtype=int)
            np.add.at(
                grid_sum,
                (x1_bin_idx[slice_mask], x2_bin_idx[slice_mask]),
                contribs,
            )
            np.add.at(
                grid_count,
                (x1_bin_idx[slice_mask], x2_bin_idx[slice_mask]),
                1,
            )
            grid = np.where(
                grid_count > 0, grid_sum / np.maximum(grid_count, 1), np.nan
            )

            im = ax.imshow(
                grid.T,
                origin="lower",
                aspect="auto",
                extent=[x1_bins[0], x1_bins[-1], x2_bins[0], x2_bins[-1]],
                cmap="RdBu_r",
            )
            plt.colorbar(im, ax=ax, label="Contribution")
            title = f"Interaction: {feature1} × {feature2}"
            if slice_label:
                title += f" | {slice_label}"
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

    used_axes = n_series * len(slice_masks)
    for ax in axes[used_axes:]:
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


def plot_interaction_heatmaps(estimator, X):
    """Shared body of the wrappers' ``plot_interactions`` method.

    Renders a binned heatmap for pairwise terms and conditioned heatmap slices
    for higher-order terms.
    """
    X_prepared, _ = prepare_plot_data(
        X,
        estimator.data_module.num_feature_info,
        estimator.data_module.cat_feature_info,
    )
    predictions = estimator._predict(X_prepared)

    interaction_names = [name for name in predictions if ":" in name]
    if not interaction_names:
        raise ValueError("No interaction terms found with predictions to plot.")

    for interaction_name in interaction_names:
        contribs = _as_2d_numpy(predictions[interaction_name])
        plot_interaction_effects(
            interaction_name,
            contribs,
            X_train_scaled=X_prepared,
            series_labels=estimator._plot_series_labels(contribs.shape[1]),
        )
