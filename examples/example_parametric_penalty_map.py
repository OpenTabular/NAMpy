"""Map two underlying smoothing parameters onto three parametric penalties."""

import numpy as np
import pandas as pd

from nampy.models import GAMRegressor


def fit_example():
    x = np.linspace(-1, 1, 100)
    frame = pd.DataFrame({"B": list(np.column_stack((x, x**2, x**3)))})
    response = .2 + .8*x + .3*x**2 - .1*x**3 + .04*np.cos(19*x)
    penalties = [np.diag(row) for row in np.eye(3)]
    mapping = np.array([[1., .5], [0., 1.], [-.5, .25]])
    underlying_sp = np.array([.4, .7])
    # Each penalty multiplier is exp(L @ log(sp)); map entries may be
    # fractional or negative. Here both underlying parameters are fixed.
    model = GAMRegressor(
        formula="y~B", optimize_smoothing=False,
        para_pen={"B": {"penalties": penalties, "L": mapping, "sp": underlying_sp}},
    ).fit(frame, response)
    return model, frame, mapping, underlying_sp


if __name__ == "__main__":
    fitted, data, L, sp = fit_example()
    print(f"Underlying smoothing parameters: {sp}")
    print(f"Full penalty multipliers: {np.exp(L @ np.log(sp))}")
    print(f"{len(fitted.predict(data))} predictions")
