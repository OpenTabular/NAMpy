"""Hurdle Poisson regression with separate intensity and presence smooths."""

import numpy as np
import pandas as pd

from nampy.gam.families.gamlss.ziplss import ZiplssFamily
from nampy.models.gamlss import GAMLSS


def fit_example():
    rng = np.random.default_rng(182)
    x = np.linspace(-1, 1, 200)
    intensity = np.exp(1 + .5 * np.sin(3 * x))
    y = rng.poisson(intensity)
    while np.any(y == 0):
        zero = y == 0
        y[zero] = rng.poisson(intensity[zero])
    present = rng.uniform(size=len(x)) < 1 - np.exp(-np.exp(.4 + x))
    y[~present] = 0
    data = pd.DataFrame({"x": x, "y": y})
    model = GAMLSS(formula=["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)"], family=ZiplssFamily()).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


if __name__ == "__main__":
    fitted, frame = fit_example()
    print("Poisson intensity/presence:", fitted.predict(frame)[:5])
    print("Expected counts:", fitted.predict_point(frame)[:5])
