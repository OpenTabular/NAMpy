#!/usr/bin/env python3
"""Cross-backend model selection with plain scikit-learn tooling.

Because both backends implement score() and sklearn tags (without mixin
classes), cross_val_score compares a GAM adapter and a neural estimator
directly.

Run:
    python examples/example_hybrid_model_selection.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from nampy.models import GAMRegressor, LinRegRegressor


def main():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"x0": rng.uniform(size=n), "x1": rng.uniform(size=n)})
    y = (
        np.sin(3.0 * X["x0"].to_numpy())
        + 0.5 * X["x1"].to_numpy()
        + rng.normal(scale=0.1, size=n)
    )

    gam = GAMRegressor(k=8)
    neural = LinRegRegressor(numerical_preprocessing="standardization")

    gam_scores = cross_val_score(gam, X, y, cv=3)
    neural_scores = cross_val_score(
        neural,
        X,
        y,
        cv=3,
        params={
            "max_epochs": 30,
            "patience": 30,
            "lr": 5e-2,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
    )

    print(f"GAMRegressor    CV R^2: {gam_scores.mean():.4f} ({gam_scores})")
    print(f"LinRegRegressor CV R^2: {neural_scores.mean():.4f} ({neural_scores})")
    winner = "GAM" if gam_scores.mean() >= neural_scores.mean() else "neural"
    print(f"Selected backend on this data: {winner}")


if __name__ == "__main__":
    main()
