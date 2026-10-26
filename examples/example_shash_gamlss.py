"""Four-parameter sinh-arcsinh additive distributional regression."""

import numpy as np
import pandas as pd
from scipy.special import ndtri

from nampy import GAMLSS
from nampy.gam.families.gamlss.shash import ShashFamily


def fit_example():
    rng = np.random.default_rng(847)
    x = np.linspace(-1.5, 1.5, 240)
    mu = 1 + .6 * np.sin(1.5 * x)
    sigma = np.exp(-.4 + .1 * x)
    epsilon, delta = .15, 1.2
    y = mu + delta * sigma * np.sinh(np.arcsinh(ndtri(rng.uniform(size=x.size))) / delta + epsilon / delta)
    data = pd.DataFrame({"x": x, "y": y})
    model = GAMLSS(
        formula={"mu": "y~s(x,bs='cr',k=6)", "sigma": "~s(x,bs='cr',k=5)",
                 "epsilon": "~1", "delta": "~1"},
        family=ShashFamily(), smoothing_optimizer="efs",
    ).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


if __name__ == "__main__":
    fitted, frame = fit_example()
    print(fitted.parameter_names_)
    print(fitted.predict(frame.head()))
    print("Conditional median:", fitted.predict_point(frame.head()))
