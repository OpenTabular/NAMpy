"""Held-out randomized residuals, worm plots and conditional count centiles.

Run with ``python examples/example_distribution_diagnostics.py OUTPUT_DIR``.
The probability/diagnostic code is identical for a supported neural LSS model.
"""

import argparse
from pathlib import Path

import matplotlib

from nampy.diagnostics import quantile_residuals

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nampy import GAMLSS
from nampy.diagnostics import centile_calibration, prepare_centiles, prepare_worm_plot
from nampy.plotting import render_centiles, render_worm_plot


def main(output_dir):
    """Fit a small count model and save its prepared diagnostics and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    x = rng.uniform(-1, 1, 600)
    mean = np.exp(1 + 0.6 * np.sin(2 * x))
    frame = pd.DataFrame({"x": x, "y": rng.negative_binomial(4, 4 / (4 + mean))})
    training, validation = frame.iloc[:400], frame.iloc[400:]
    model = GAMLSS(
        family="nb_lss",
        formula={"mu": "y ~ s(x, bs='cr', k=7)", "alpha": "~ 1"},
        optimize_smoothing=False,
        smoothing_params=[0.5],
    ).fit(training)
    distribution = model.predict_distribution(validation)
    residuals = quantile_residuals(distribution, validation.y, random_state=7)
    worm = prepare_worm_plot(residuals, x=validation.x, n_bins=4)
    worm_figure = render_worm_plot(worm)
    worm_figure.savefig(output_dir / "worm.png", dpi=150)

    grid = pd.DataFrame({"x": np.linspace(-1, 1, 150)})
    curves = prepare_centiles(
        model.predict_distribution(grid), grid.x, probs=[0.1, 0.5, 0.9]
    )
    centile_figure = render_centiles(curves, xlabel="x", ylabel="Count")
    centile_figure.savefig(output_dir / "centiles.png", dpi=150)
    calibration = centile_calibration(distribution, validation.y, probs=[0.1, 0.5, 0.9])
    pd.DataFrame(
        {
            "probability": calibration.probabilities,
            "observed_coverage": calibration.observed,
            "attainable_model_probability": calibration.model_probability,
            "n_used": calibration.n_used,
        }
    ).to_csv(output_dir / "calibration.csv", index=False)
    pd.DataFrame(
        {
            "y": validation.y.to_numpy(),
            "pit": residuals.pit,
            "residual": residuals.residuals,
            "randomized": residuals.randomized,
            "clipped": residuals.clipped,
        }
    ).to_csv(output_dir / "residuals.csv", index=False)
    plt.close(worm_figure)
    plt.close(centile_figure)
    return calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.output_dir)
