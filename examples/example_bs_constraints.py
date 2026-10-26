"""Fit a derivative-penalized B-spline with an explicit coefficient constraint."""

import numpy as np
import pandas as pd

from nampy.models import GAMRegressor


def fit_example():
    x = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"x": x})
    response = np.sin(5 * x) + .08 * np.cos(23 * x)
    # C acts on all eight uncentered B-spline coefficients. This replaces the
    # default data-dependent centering row with a coefficient-sum constraint.
    constraint = np.ones((1, 8))
    model = GAMRegressor(
        formula="y~s(x,bs='bs',k=8,m=[3,2])",
        constraints={1: constraint},
        smoothing_params=[.7],
        optimize_smoothing=False,
    ).fit(frame, response)
    return model, frame, constraint


if __name__ == "__main__":
    fitted, data, _ = fit_example()
    predicted = fitted.predict(data)
    print(f"{len(predicted)} predictions, finite={bool(np.isfinite(predicted).all())}")
