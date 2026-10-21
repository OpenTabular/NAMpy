"""Backend-neutral rendering of prepared term-plot data."""

from .distributional import (
    DistributionalEffects,
    add_density_shading,
    combine_distributional_effects,
    plot_distributional_effects,
    prepare_distributional_effects,
    render_distributional_effects,
)
from .render import prepared_from_contributions, render_term_plots

__all__ = [
    "DistributionalEffects",
    "add_density_shading",
    "combine_distributional_effects",
    "plot_distributional_effects",
    "prepare_distributional_effects",
    "prepared_from_contributions",
    "render_distributional_effects",
    "render_term_plots",
]
