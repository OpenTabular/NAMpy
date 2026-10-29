"""Generate the light and dark NAM feature-effect plots used on the homepage."""

from dataclasses import replace
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.configuration import GradientTrainingConfig
from nampy.neural.configuration import PreprocessingConfig

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nampy.models import NAMRegressor


def _style_figure(fig, *, dark: bool) -> None:
    """Apply documentation colors without changing the plotted quantities."""
    background = "#0f172a" if dark else "#ffffff"
    foreground = "#e5e7eb" if dark else "#111827"
    grid = "#334155" if dark else "#e5e7eb"
    effect = "#60a5fa" if dark else "#2563eb"
    observations = "#94a3b8" if dark else "#64748b"

    fig.patch.set_facecolor(background)
    for ax in fig.axes:
        ax.set_facecolor(background)
        ax.grid(color=grid, linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(colors=foreground)
        ax.xaxis.label.set_color(foreground)
        ax.yaxis.label.set_color(foreground)
        ax.title.set_color(foreground)
        for spine in ax.spines.values():
            spine.set_color(grid)

        if ax.lines:
            ax.lines[0].set_color(effect)
            ax.lines[0].set_linewidth(2.4)
        for line in ax.lines[1:]:
            line.set_color(observations)
            line.set_alpha(0.28)


def main() -> None:
    rng = np.random.default_rng(7)
    n = 1_800
    X = pd.DataFrame(
        {
            "median_income": rng.uniform(1.0, 10.0, n),
            "house_age": rng.uniform(1.0, 52.0, n),
            "latitude": rng.uniform(32.0, 42.0, n),
            "rooms_per_household": rng.lognormal(1.5, 0.35, n),
        }
    )
    y = (
        2.2 * np.tanh((X["median_income"].to_numpy() - 5.0) / 1.7)
        + 0.9 * np.sin((X["house_age"].to_numpy() - 5.0) / 8.0)
        - 0.55 * (X["latitude"].to_numpy() - 37.0) ** 2 / 6.0
        + 0.75 * np.log1p(X["rooms_per_household"].to_numpy())
        + rng.normal(0.0, 0.3, n)
    )

    model = NAMRegressor(architecture_config=DefaultNAMConfig(layer_sizes=[48, 24], dropout=0.0), preprocessing=PreprocessingConfig(numerical_preprocessing='minmax'))
    with TemporaryDirectory() as checkpoint_dir:
        model.set_params(training=replace(model.training or GradientTrainingConfig(), max_epochs=100, batch_size=128, patience=12, random_state=7, checkpoint_path=checkpoint_dir, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False})).fit(X, y)

    output_dir = Path(__file__).resolve().parents[1] / "docs" / "_static"
    display_names = {
        "median_income": "Median income",
        "house_age": "House age",
        "latitude": "Latitude",
        "rooms_per_household": "Rooms per household",
    }
    for dark, filename in (
        (False, "homepage_demo_light.png"),
        (True, "homepage_demo_dark.png"),
    ):
        figures = model.plot_terms(
            X,
            center=True,
            reference_X=X,
            pages=1,
            figsize=(10.5, 7.8),
        )
        fig = figures[0]
        for ax, label in zip(fig.axes, display_names.values(), strict=True):
            ax.set_title(label)
            ax.set_xlabel("Feature value")
            ax.set_ylabel(f"Effect of {label.lower()}")
        fig.subplots_adjust(hspace=0.42, wspace=0.18)
        _style_figure(fig, dark=dark)
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
