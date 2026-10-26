"""Gumbel location/scale regression with two smooth predictors."""

import numpy as np
import pandas as pd

from nampy.gam.families.gamlss.gumbls import GumblsFamily
from nampy.models.gamlss import GAMLSS


def fit_example():
    rng = np.random.default_rng(181)
    x = np.linspace(-1, 1, 160)
    y = rng.gumbel(np.sin(3 * x), np.exp(-.4 + .2 * x))
    data = pd.DataFrame({"x": x, "y": y})
    model = GAMLSS(formula=["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)"], family=GumblsFamily()).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


if __name__ == "__main__":
    fitted, frame = fit_example()
    print("Gumbel location/scale:", fitted.predict(frame)[:5])
    print("Expected response:", fitted.predict_point(frame)[:5])
