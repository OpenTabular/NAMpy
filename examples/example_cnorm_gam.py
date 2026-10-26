"""Censored normal location GAM with fixed SD and smoothing."""

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.families.cnorm import CensoredNormalFamily


def example_data():
    x = np.linspace(-1, 1, 120)
    latent = 3 + np.sin(3 * x) + np.random.default_rng(731).normal(size=len(x))
    value, bound = latent.copy(), latent.copy()
    # A detection limit supplies a threshold, not the hidden latent value.
    left = (np.arange(len(x)) % 4 == 1) & (latent < 3)
    right = (np.arange(len(x)) % 4 == 2) & (latent > 3)
    interval = np.arange(len(x)) % 4 == 3
    value[left], bound[left] = 3, -np.inf
    value[right], bound[right] = 3, np.inf
    value[interval] = np.floor(latent[interval] * 2) / 2
    bound[interval] = value[interval] + 0.5
    return pd.DataFrame({"x": x, "value": value, "bound": bound})


def fit_example():
    return GAM(
        formula="cbind(value, bound) ~ s(x, bs='cr', k=7)",
        family=CensoredNormalFamily(theta=1.0, link="identity"),
        smoothing_params=[0.7],
        optimize_smoothing=False,
    ).fit(data=example_data())


if __name__ == "__main__":
    model = fit_example()
    location, se = model.predict(type="response", return_se=True)
    print("Fixed response SD:", model.family.getTheta(trans=True))
    print("First fitted locations:", location[:5])
    print("First location standard errors:", se[:5])
