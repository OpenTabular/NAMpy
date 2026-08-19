"""GAMResidual plot(): both backends' 1-d curves render via the renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nampy.hybrid import GAMResidualRegressor
from nampy.models.linreg import LinRegRegressor

_KW = {
    "max_epochs": 3,
    "patience": 3,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def test_plot_renders_gam_and_neural_curves(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    data = pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)})
    data["y"] = (
        np.sin(3.0 * data["x0"]) + data["x3"]
        + rng.normal(scale=0.1, size=n)
    )

    estimator = GAMResidualRegressor(
        "y ~ s(x0, k=5)",
        LinRegRegressor(numerical_preprocessing="standardization"),
    )
    estimator.fit(
        data,
        neural_features=["x3"],
        neural_fit_kwargs=dict(_KW, checkpoint_path=str(tmp_path)),
    )

    figures = estimator.plot(data)
    try:
        labels = [ax.get_xlabel() for fig in figures for ax in fig.axes]
        # The GAM smooth renders with its prefixed formula label mapped to
        # the x0 column; the neural term keeps its prefixed feature label.
        assert any(label.startswith("gam:s(x0") for label in labels)
        assert "nn:x3" in labels
    finally:
        plt.close("all")
