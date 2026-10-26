"""Censored Poisson mean with explicit paired observation bounds."""

import numpy as np
import pandas as pd

from nampy.models import GAMRegressor


def fit_example():
    rng = np.random.default_rng(764)
    x = np.linspace(-1, 1, 80)
    y = rng.poisson(np.exp(1 + .5 * np.sin(2.5 * x))) + 1
    bound = y.astype(float)
    bound[1::4] = -np.inf
    bound[2::4] = np.inf
    bound[3::4] += 2
    data = pd.DataFrame({"x": x, "y": y, "bound": bound})
    model = GAMRegressor(
        family="cpois", formula="cbind(y,bound) ~ s(x,bs='cr',k=7)",
        smoothing_method="reml",
    ).fit(data)
    return model, data


if __name__ == "__main__":
    model, data = fit_example()
    print(model.predict(data)[:5])
