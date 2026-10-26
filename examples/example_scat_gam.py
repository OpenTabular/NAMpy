"""Scaled Student t GAM with fixed degrees of freedom, scale, and smoothing."""

import numpy as np
import pandas as pd

from nampy.gam import GAM


def fit_example():
    rng = np.random.default_rng(327)
    x = np.linspace(-1, 1, 100)
    y = 0.6 * np.sin(3 * x) + 0.3 * rng.standard_t(5, len(x))
    data = pd.DataFrame({"x": x, "y": y})
    return GAM(
        formula="y ~ s(x, bs='cr', k=7)",
        family={"name": "scat", "theta": [5, 0.3]},
        smoothing_params=[0.4],
        optimize_smoothing=False,
    ).fit(data=data)


if __name__ == "__main__":
    model = fit_example()
    mean, se = model.predict(return_se=True)
    print("Degrees of freedom and scale:", model.family.getTheta(trans=True))
    print("First fitted locations:", mean[:5])
    print("First standard errors:", se[:5])
