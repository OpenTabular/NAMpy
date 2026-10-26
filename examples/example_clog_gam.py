"""Logistic location GAM with exact, interval and one-sided observations."""

import numpy as np
import pandas as pd

from nampy.models import GAMRegressor


def example_data():
    x = np.linspace(-1, 1, 160)
    latent = 3 + np.sin(3 * x) + np.random.default_rng(732).logistic(scale=0.7, size=x.size)
    value, bound = latent.copy(), latent.copy()
    left = (np.arange(x.size) % 4 == 1) & (latent < 3)
    right = (np.arange(x.size) % 4 == 2) & (latent > 3)
    interval = np.arange(x.size) % 4 == 3
    value[left], bound[left] = 3, -np.inf
    value[right], bound[right] = 3, np.inf
    value[interval] = np.floor(latent[interval] * 2) / 2
    bound[interval] = value[interval] + 0.5
    return pd.DataFrame({"x": x, "value": value, "bound": bound})


def fit_example():
    return GAMRegressor(
        formula="cbind(value, bound) ~ s(x, bs='cr', k=7)",
        family={"name": "clog", "theta": 0.7},
        smoothing_params=[0.7],
        optimize_smoothing=False,
    ).fit(example_data())


def fit_estimated_example():
    """Jointly estimate logistic scale and smoothing with source EFS updates."""
    return GAMRegressor(
        formula="cbind(value, bound) ~ s(x, bs='cr', k=7)",
        family="clog",
        smoothing_method="REML",
        smoothing_optimizer="efs",
    ).fit(example_data())


if __name__ == "__main__":
    model = fit_example()
    print("First logistic locations:", model.predict(example_data()[["x"]])[:5])
    print("Fixed logistic scale:", model.gam_.family.getTheta(trans=True))
    estimated = fit_estimated_example()
    print("Estimated logistic scale:", estimated.gam_.family.getTheta(trans=True))
