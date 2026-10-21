"""Generate the distributional feature-effect plot shown in the README."""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nampy.models import NAMLSS


def main() -> None:
    rng = np.random.default_rng(11)
    n = 1_800
    X = pd.DataFrame(
        {
            "median_income": rng.uniform(1.0, 10.0, n),
            "house_age": rng.uniform(1.0, 52.0, n),
        }
    )
    location = (
        1.8 * np.tanh((X["median_income"].to_numpy() - 5.0) / 1.7)
        + 0.7 * np.sin((X["house_age"].to_numpy() - 5.0) / 8.0)
    )
    scale = 0.25 + 0.45 / (
        1.0 + np.exp(-(X["median_income"].to_numpy() - 5.0))
    )
    y = location + rng.normal(0.0, scale, n)

    model = NAMLSS(
        family="normal",
        layer_sizes=[48, 24],
        dropout=0.0,
        numerical_preprocessing="minmax",
    )
    with TemporaryDirectory() as checkpoint_dir:
        model.fit(
            X,
            y,
            max_epochs=100,
            batch_size=128,
            patience=12,
            random_state=11,
            checkpoint_path=checkpoint_dir,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    figures = model.plot_distributional_effects(
        X,
        reference_X=X,
        feature_names=["median_income", "house_age"],
        display_names={
            "median_income": "Median income",
            "house_age": "House age",
        },
        ncols=2,
        figsize=(10.5, 3.5),
    )
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "_static"
        / "readme_distributional_effects.png"
    )
    figures[0].savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figures[0])


if __name__ == "__main__":
    main()
