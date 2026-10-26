"""Single-predictor hurdle Poisson GAM with fixed ancillary parameters."""

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.families import ZeroInflatedPoissonFamily


def fit_example():
    x = np.linspace(-1, 1, 180)
    gamma = 0.2 + 1.2 * np.sin(3 * x) + 0.2 * x
    family = ZeroInflatedPoissonFamily(theta=[-0.5, np.log(0.5)], b=0.3)
    y = family.rd(gamma, rng=np.random.default_rng(327))
    data = pd.DataFrame({"x": x, "y": y})
    return GAM(
        formula="y ~ s(x, bs='cr', k=7)",
        family=family,
        smoothing_params=[0.4],
        optimize_smoothing=False,
    ).fit(data=data)


if __name__ == "__main__":
    model = fit_example()
    mean, se = model.predict(type="response", return_se=True)
    print("Presence intercept and slope:", model.family.getTheta(trans=True))
    print("First expected counts:", mean[:5])
    print("First mean standard errors:", se[:5])
