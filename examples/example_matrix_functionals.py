"""Weighted surface and tensor functionals stored as array-valued columns."""

import numpy as np
import pandas as pd

from nampy.gam import GAM


def fit_examples():
    rng = np.random.default_rng(219)
    x = rng.uniform(-1, 1, (90, 4))
    z = rng.uniform(-1, 1, x.shape)
    weights = rng.uniform(0.1, 0.6, x.shape)
    response = 0.4 + np.sum(weights * (np.sin(2 * x) + 0.3 * z), axis=1)
    response += rng.normal(0, 0.1, len(response))
    data = pd.DataFrame({"y": response, "X": list(x), "Z": list(z), "L": list(weights)})
    surface = GAM(
        formula="y ~ s(X,Z,by=L,bs='tp',k=12)",
        smoothing_method="REML", optimize_smoothing=True,
    ).fit(data=data)
    tensor = GAM(
        formula="y ~ te(X,Z,by=L,bs=['cr','cr'],k=[5,5])",
        smoothing_method="REML", optimize_smoothing=True,
    ).fit(data=data)
    return data, surface, tensor


if __name__ == "__main__":
    data, surface, tensor = fit_examples()
    for model in (surface, tensor):
        fit, se = model.predict(data.iloc[:5], return_se=True)
        print(pd.DataFrame({"predicted_mean": fit, "standard_error": se}))
