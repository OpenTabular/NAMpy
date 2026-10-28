"""Backend-neutral preparation of observed interaction contribution slices."""

import numpy as np
import pandas as pd


def _interaction_panels(
    frame,
    features,
    interaction_preds,
    *,
    num_bins,
    slice_bins,
    max_slices,
    series_labels,
):
    features = list(features)
    if len(features) < 2:
        raise ValueError("An interaction plot requires at least two features.")
    feature1, feature2 = features[:2]

    interaction_preds = np.asarray(interaction_preds).reshape(len(frame), -1)
    n_series = interaction_preds.shape[1]

    x1_vals = frame[feature1].to_numpy()
    x2_vals = frame[feature2].to_numpy()

    x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
    x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
    x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
    x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

    slice_masks = [("", np.ones(len(interaction_preds), dtype=bool))]
    if len(features) > 2:
        conditions = pd.DataFrame(index=np.arange(len(interaction_preds)))
        for feature in features[2:]:
            values = frame[feature]
            if pd.api.types.is_numeric_dtype(values):
                conditions[feature] = (
                    pd.qcut(values, q=slice_bins, duplicates="drop")
                    .astype(str)
                    .to_numpy()
                )
            else:
                conditions[feature] = values.astype(str).to_numpy()
        keys = list(conditions)
        grouped = conditions.groupby(
            keys[0] if len(keys) == 1 else keys, dropna=False
        ).groups
        ranked_groups = sorted(
            grouped.items(), key=lambda item: (-len(item[1]), str(item[0]))
        )[:max_slices]
        slice_masks = []
        for condition, row_indices in ranked_groups:
            values = condition if isinstance(condition, tuple) else (condition,)
            label = ", ".join(
                f"{name}={value}"
                for name, value in zip(features[2:], values, strict=True)
            )
            mask = np.zeros(len(interaction_preds), dtype=bool)
            mask[np.asarray(row_indices, dtype=int)] = True
            slice_masks.append((label, mask))

    panels = []
    for slice_label, slice_mask in slice_masks:
        for out_idx in range(n_series):
            grid_sum = np.zeros((num_bins - 1, num_bins - 1))
            grid_count = np.zeros((num_bins - 1, num_bins - 1), dtype=int)
            indices = (x1_bin_idx[slice_mask], x2_bin_idx[slice_mask])
            np.add.at(grid_sum, indices, interaction_preds[slice_mask, out_idx])
            np.add.at(grid_count, indices, 1)
            grid = np.where(
                grid_count > 0, grid_sum / np.maximum(grid_count, 1), np.nan
            )
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
            panels.append(
                {
                    "kind": "2d",
                    "plot_me": True,
                    "scheme": 2,
                    "n2": num_bins - 1,
                    "fit": grid.T.ravel(),
                    "x": np.linspace(x1_bins[0], x1_bins[-1], num_bins - 1),
                    "y": np.linspace(x2_bins[0], x2_bins[-1], num_bins - 1),
                    "xlab": feature1,
                    "ylab": feature2,
                    "main": title,
                    "contours": False,
                    "colorbar": True,
                    "cmap": "RdBu_r",
                }
            )
    return panels


def prepared_from_interactions(
    frame,
    terms,
    *,
    term_features=None,
    series_labels: list[str] | tuple[str, ...] | None = None,
    num_bins=30,
    slice_bins=3,
    max_slices=9,
):
    """Bin observed contributions, conditioning higher-order terms on remaining features."""
    if num_bins < 3 or slice_bins < 1 or max_slices < 1:
        raise ValueError(
            "num_bins must be at least 3; slice_bins and max_slices must be positive."
        )
    term_features = dict(term_features or {})
    panels = []
    for name, values in terms.items():
        features = term_features.get(name, tuple(name.split(":")))
        if len(features) < 2:
            continue
        panels.extend(
            _interaction_panels(
                frame,
                features,
                values,
                num_bins=num_bins,
                slice_bins=slice_bins,
                max_slices=max_slices,
                series_labels=series_labels,
            )
        )
    if not panels:
        raise ValueError("No interaction terms found with predictions to plot.")
    return {
        "pd": panels,
        "ylim": None,
        "partial_resids": False,
        "shift": 0.0,
        "trans": lambda values: values,
        "select": None,
        "rug_default": False,
    }
