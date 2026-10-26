"""Soap functionals with a factor that varies over quadrature locations."""

import numpy as np
import pandas as pd

from nampy.models import GAMRegressor


def fit_example():
    rng = np.random.default_rng(463)
    n, width = 120, 3
    X, Y = rng.uniform(-.8, .8, (2, n, width))
    L = rng.uniform(.2, 1, (n, width))
    codes = (np.arange(n)[:, None] + np.arange(width)) % 3
    levels = ["a", "b", "c"]
    labels = np.asarray(levels)[codes]
    frame = pd.DataFrame({"X": list(X), "Y": list(Y), "L": list(L)})
    # Each row carries the declared factor levels as well as its observations.
    # Ordinary arrays of strings have no categorical metadata.
    frame["group"] = pd.Series(
        [pd.Categorical(row, categories=levels) for row in labels], dtype=object
    )
    response = .2 + np.sum(L * (.4 + .2 * codes) * (np.sin(X) + .2 * Y), axis=1)
    response += rng.normal(0, .05, n)
    boundary = [{"X": [-1., 1., 1., -1., -1.],
                 "Y": [-1., -1., 1., 1., -1.]}]
    model = GAMRegressor(
        formula=f"y~s(X,Y,group,bs='fs',k=8,by=L,xt={{'bs':'sf','bnd':{boundary!r},'nmax':30}})",
        knots={"X": [-.5, .5, -.5, .5], "Y": [-.5, -.5, .5, .5]},
        smoothing_params=[.3, .3], optimize_smoothing=False,
    ).fit(frame, response)
    return model, frame


if __name__ == "__main__":
    fitted, data = fit_example()
    predicted = fitted.predict(data)
    print(f"{len(predicted)} predictions, finite={bool(np.isfinite(predicted).all())}")
