"""Censored Box-Cox Gaussian model with lambda and SD estimated jointly."""

import numpy as np
import pandas as pd

from nampy.gam.families.bcg import CensoredBoxCoxGaussianFamily
from nampy.models.gam import GAMRegressor


def fit_example():
    rng = np.random.default_rng(183)
    x = np.linspace(-1, 1, 120)
    y = np.exp(.7 + .25 * np.sin(3 * x) + rng.normal(0, .18, len(x)))
    bound = y.copy()
    bound[1::10] = 0.
    bound[2::10] = np.inf
    bound[3::10] += .25
    data = pd.DataFrame({"x": x, "y": y, "bound": bound})
    model = GAMRegressor(formula="cbind(y,bound)~s(x,bs='cr',k=6)", family=CensoredBoxCoxGaussianFamily()).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


if __name__ == "__main__":
    fitted, frame = fit_example()
    print("Box-Cox lambda and SD:", fitted.gam_.family.getTheta(True))
    print("Transformed-location predictions:", fitted.predict(frame)[:5])
