"""Tweedie mean, bounded power and dispersion additive predictors."""

import numpy as np
import pandas as pd

from nampy import GAMLSS
from nampy.gam.families.gamlss.twlss import TwlssFamily


def fit_example():
    rng = np.random.default_rng(2493)
    x = np.linspace(-1.5, 1.5, 180)
    mu = np.exp(.4 + .4 * np.sin(1.6 * x))
    p, phi = 1.5, .6
    counts = rng.poisson(mu**(2 - p) / (phi * (2 - p)))
    y = np.zeros(x.size)
    positive = counts > 0
    y[positive] = rng.gamma(counts[positive] * (2 - p) / (p - 1),
                            phi * (p - 1) * mu[positive]**(p - 1))
    data = pd.DataFrame({"x": x, "y": y})
    model = GAMLSS(
        formula={"mu": "y~s(x,bs='cr',k=6)", "p": "~1", "phi": "~s(x,bs='cr',k=5)"},
        family=TwlssFamily(), smoothing_optimizer="efs",
    ).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


if __name__ == "__main__":
    fitted, frame = fit_example()
    print(fitted.parameter_names_)
    print(fitted.predict(frame.head()))
    print("Conditional mean:", fitted.predict_point(frame.head()))
