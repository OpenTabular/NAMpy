"""Shared plotting utilities for NAMpy models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_grid_layout(n_plots, max_cols=4):
    """
    Compute optimal grid dimensions for a given number of plots.

    Parameters
    ----------
    n_plots : int
        Number of plots to arrange in a grid.
    max_cols : int, optional
        Maximum number of columns, by default 4.

    Returns
    -------
    tuple
        (nrows, ncols) dimensions for the grid.
    """
    if n_plots <= 0:
        return 0, 0
    ncols = min(n_plots, max_cols)
    nrows = int(np.ceil(n_plots / ncols))
    return nrows, ncols


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
    nrows, ncols = compute_grid_layout(n_plots, max_cols)
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
