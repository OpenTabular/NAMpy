"""Backend-neutral rendering of prepared term-plot data."""

from .checks import render_model_check, render_qq
from .distributional import (
    DistributionalEffects,
    add_density_shading,
    combine_distributional_effects,
    plot_distributional_effects,
    prepare_distributional_effects,
    render_distributional_effects,
)
from .interactions import prepared_from_interactions
from .render import prepared_from_contributions, render_term_plots
from .surfaces import render_prediction_surface

__all__ = [
    "render_model_check",
    "render_qq",
    "DistributionalEffects",
    "add_density_shading",
    "combine_distributional_effects",
    "plot_distributional_effects",
    "prepare_distributional_effects",
    "prepared_from_contributions",
    "prepared_from_interactions",
    "render_distributional_effects",
    "render_term_plots",
    "render_prediction_surface",
]
