"""Family-agnostic feature-effect plots for distributional additive models."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..inspection import effects as effect_preparation


def add_density_shading(
    ax,
    values,
    *,
    bins: int = 30,
    color: str = "#D73027",
    min_alpha: float = 0.03,
    max_alpha: float = 0.28,
    label: str | None = None,
) -> None:
    """Shade vertical bins according to the empirical predictor density."""
    if bins < 1:
        raise ValueError("bins must be positive.")
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        return
    counts, edges = np.histogram(x, bins=np.linspace(lo, hi, bins + 1))
    maximum = max(int(counts.max()), 1)
    for index, count in enumerate(counts):
        alpha = min_alpha + (max_alpha - min_alpha) * float(count) / maximum
        ax.axvspan(
            edges[index],
            edges[index + 1],
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=0,
            label=label if index == 0 else None,
        )


def render_distributional_effects(
    effects: effect_preparation.DistributionalEffects,
    *,
    feature_names: Sequence[str] | None = None,
    display_names: Mapping[str, str] | None = None,
    effect_bins: int = 20,
    density_bins: int = 24,
    quantiles: tuple[float, float] = (0.1, 0.9),
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    point_label: str = "point effect",
    envelope_label: str = "point $\\pm$ conditional SD",
    fit_label: str = "fit",
    ylabel: str = "Point effect and conditional SD",
    title: str | None = None,
):
    """Render Figure-5-style effects from one or several fitted curves.

    Blue is the median point effect, black is point effect plus/minus one
    conditional standard deviation, red intensity is feature density, and the
    blue ribbon (when multiple fits are supplied) spans ``quantiles``.
    """
    import matplotlib.pyplot as plt

    if effect_bins < 1 or density_bins < 1 or ncols < 1:
        raise ValueError("effect_bins, density_bins, and ncols must be positive.")
    lower_q, upper_q = map(float, quantiles)
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")

    terms = list(effects.point_effects)
    if feature_names is not None:
        unknown = set(feature_names) - set(terms)
        if unknown:
            raise ValueError(f"Unknown feature terms: {sorted(unknown)}")
        terms = list(feature_names)
    if not terms:
        raise ValueError("feature_names must select at least one term.")
    names = dict(display_names or {})
    columns = min(ncols, len(terms))
    rows = math.ceil(len(terms) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize or (5.4 * columns, 3.05 * rows),
        squeeze=False,
    )

    for index, term in enumerate(terms):
        ax = axes.flat[index]
        x = effects.feature_values[term]
        point = effects.point_effects[term]
        spread = effects.conditional_sd[term]
        centers, point_curves, spread_curves = effect_preparation._binned_effects(
            x, point, spread, bins=effect_bins
        )
        point_median = np.nanmedian(point_curves, axis=1)
        spread_median = np.nanmedian(spread_curves, axis=1)
        plot_min, plot_max = float(np.min(centers)), float(np.max(centers))
        supported_x = x[(x >= plot_min) & (x <= plot_max)]
        add_density_shading(
            ax,
            supported_x,
            bins=density_bins,
            label="feature density" if index == 0 else None,
        )
        ax.plot(
            centers,
            point_median - spread_median,
            color="black",
            linewidth=0.9,
            zorder=2,
            label=envelope_label if index == 0 else None,
        )
        ax.plot(
            centers,
            point_median + spread_median,
            color="black",
            linewidth=0.9,
            zorder=2,
        )
        ax.plot(
            centers,
            point_median,
            color="#276FBF",
            linewidth=2.0,
            zorder=4,
            label=point_label if index == 0 else None,
        )
        if effects.n_fits > 1:
            ax.fill_between(
                centers,
                np.nanquantile(point_curves, lower_q, axis=1),
                np.nanquantile(point_curves, upper_q, axis=1),
                color="#276FBF",
                alpha=0.16,
                linewidth=0,
                zorder=3,
                label=(
                    f"{fit_label} {100 * lower_q:g}–{100 * upper_q:g}%"
                    if index == 0
                    else None
                ),
            )
        ax.set_title(names.get(term, term))
        ax.set_xlabel("Feature value")
        ax.set_xlim(plot_min, plot_max)
        if index % columns == 0:
            ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
        ax.grid(axis="x", alpha=0.25)

    for ax in axes.flat[len(terms) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, frameon=False, fontsize=8)
    if title:
        figure.suptitle(title, fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.tight_layout()
    return [figure]


def plot_distributional_effects(estimator, X, **kwargs):
    """Prepare and render distributional effects for one fitted estimator.

    Preparation keywords are ``reference_X``, ``reference_weight``, and
    ``response_scale``.  Remaining keywords are passed to
    :func:`render_distributional_effects`.
    """
    prepare_keys = {"reference_X", "reference_weight", "response_scale"}
    prepare_kwargs = {
        name: kwargs.pop(name) for name in tuple(kwargs) if name in prepare_keys
    }
    feature_name = kwargs.pop("feature_name", None)
    feature_names = kwargs.get("feature_names")
    if feature_name is not None:
        if feature_names is not None:
            raise ValueError("Use either feature_name or feature_names, not both.")
        feature_names = [feature_name]
        kwargs["feature_names"] = feature_names
    effects = effect_preparation.prepare_distributional_effects(
        estimator, X, feature_names=feature_names, **prepare_kwargs
    )
    return render_distributional_effects(effects, **kwargs)


__all__ = [
    "add_density_shading",
    "plot_distributional_effects",
    "render_distributional_effects",
]
