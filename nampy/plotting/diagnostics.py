"""Render shared distribution diagnostics and conditional centile curves."""

from numbers import Integral

import numpy as np

from ..diagnostics.distribution import prepare_centiles, prepare_worm_plot


def render_worm_plot(data, *, columns=2, figsize=None):
    """Return a Figure from WormPlotData, preserving empty/small-group status.

    Shading is a pointwise normal reference envelope, not a confidence band
    accounting for model estimation. Missing/infinite residual counts and
    ungrouped rows are displayed explicitly.
    """
    import matplotlib.pyplot as plt

    if isinstance(columns, bool) or not isinstance(columns, Integral) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    n_panels = max(1, len(data.panels))
    columns = min(columns, n_panels)
    rows = (n_panels + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=figsize or (5 * columns, 3.8 * rows)
    )
    for ax, panel in zip(axes.flat, data.panels, strict=False):
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.fill_between(
            panel.theoretical,
            panel.envelope_lower,
            panel.envelope_upper,
            alpha=0.2,
            label=f"{data.reference_level:.0%} pointwise normal reference",
        )
        ax.scatter(panel.theoretical, panel.deviation, s=14, color="black")
        if panel.cubic_coefficients is not None:
            ax.plot(
                panel.theoretical,
                np.polynomial.polynomial.polyval(
                    panel.theoretical, panel.cubic_coefficients
                ),
                color="tab:red",
                label="Cubic trend",
            )
        else:
            ax.text(
                0.02,
                0.97,
                "Cubic trend unavailable"
                if len(panel.theoretical) < 4
                else "Cubic trend disabled",
                transform=ax.transAxes,
                va="top",
            )
        ax.set(
            xlabel="Standard normal quantile",
            ylabel="Residual minus normal quantile",
            title=f"{panel.label}\nn={len(panel.theoretical)}/{panel.n_total}; missing={panel.n_missing}, infinite={panel.n_infinite}",
        )
        ax.title.set_fontsize(10)
        ax.legend(loc="best", fontsize="small")
    for ax in list(axes.flat)[len(data.panels) :]:
        ax.set_visible(False)
    if not data.panels:
        axes[0, 0].set_visible(True)
        axes[0, 0].set_axis_off()
        axes[0, 0].text(
            0.5,
            0.5,
            "No grouped residuals",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
    fig.suptitle(f"Worm plot; ungrouped rows={data.n_ungrouped}")
    fig.tight_layout()
    return fig


def plot_worm(
    residuals,
    *,
    groups=None,
    x=None,
    n_bins=4,
    breaks=None,
    fit_cubic=True,
    reference_level=0.95,
    columns=2,
    figsize=None,
):
    """Prepare and render a shared worm plot; returns a Matplotlib Figure."""
    data = prepare_worm_plot(
        residuals,
        groups=groups,
        x=x,
        n_bins=n_bins,
        breaks=breaks,
        fit_cubic=fit_cubic,
        reference_level=reference_level,
    )
    return render_worm_plot(data, columns=columns, figsize=figsize)


def render_centiles(
    data,
    *,
    ax=None,
    xlabel="Covariate",
    ylabel="Response",
    title="Conditional centiles",
):
    """Return a Figure from CentileCurves, without uncertainty-band shading."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for j, probability in enumerate(data.probabilities):
        ax.plot(
            data.x,
            data.quantiles[:, j],
            label=f"{100 * probability:g}%",
            drawstyle="steps-post" if data.response_kind == "discrete" else "default",
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(title="Conditional quantile")
    ax.figure.tight_layout()
    return ax.figure


def plot_centiles(
    distribution,
    x,
    *,
    probs=(0.05, 0.5, 0.95),
    ax=None,
    xlabel="Covariate",
    ylabel="Response",
    title="Conditional centiles",
):
    """Prepare and render conditional centiles at supplied rows; returns a Figure."""
    return render_centiles(
        prepare_centiles(distribution, x, probs=probs),
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
