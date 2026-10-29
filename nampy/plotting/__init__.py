"""Backend-neutral rendering of prepared term-plot data."""

from .checks import render_model_check, render_qq
from .diagnostics import plot_centiles, plot_worm, render_centiles, render_worm_plot
from .effects import (
    add_density_shading,
    plot_distributional_effects,
    render_distributional_effects,
)
from .interactions import prepared_from_interactions
from .render import prepared_from_contributions, render_term_plots
from .surfaces import render_prediction_surface

__all__ = [
    "plot_centiles",
    "plot_worm",
    "render_centiles",
    "render_worm_plot",
    "render_model_check",
    "render_qq",
    "add_density_shading",
    "plot_distributional_effects",
    "prepared_from_contributions",
    "prepared_from_interactions",
    "render_distributional_effects",
    "render_term_plots",
    "render_prediction_surface",
]
