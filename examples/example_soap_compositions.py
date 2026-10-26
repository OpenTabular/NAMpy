"""Known-boundary soap tensors and matrix-valued soap factor smooths."""

import numpy as np
import pandas as pd

from nampy.gam import GAM


def fit_examples():
    rng = np.random.default_rng(415)
    n, width = 120, 3
    x, y = rng.uniform(-.8, .8, (2, n))
    z = rng.uniform(-1, 1, n)
    boundary = [{"x": [-1., 1., 1., -1., -1.],
                 "y": [-1., -1., 1., 1., -1.]}]
    known = [{**boundary[0], "f": [0.] * 5}]
    knots = {"x": [-.5, .5, -.5, .5], "y": [-.5, -.5, .5, .5]}
    frame = pd.DataFrame({"x": x, "y": y, "z": z,
                          "response": .3 + np.sin(x) + .3 * y + .2 * z
                          + rng.normal(0, .05, n)})
    tensor = GAM(
        formula=f"response~te(x,y,z,bs=['so','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{known!r},'nmax':30}},None])",
        knots=knots, smoothing_params=[.3, .3], optimize_smoothing=False,
    ).fit(data=frame)

    X, Y = rng.uniform(-.8, .8, (2, n, width))
    L = rng.uniform(.2, 1, (n, width))
    factor = pd.Categorical(np.resize(["a", "b", "c"], n))
    amplitude = .4 + .2 * factor.codes
    matrix = pd.DataFrame({"X": list(X), "Y": list(Y), "L": list(L), "group": factor,
                           "response": .2 + amplitude * np.sum(L * (np.sin(X) + .2 * Y), axis=1)
                           + rng.normal(0, .05, n)})
    renamed_boundary = [{"X": boundary[0]["x"], "Y": boundary[0]["y"]}]
    grouped = GAM(
        formula=f"response~s(X,Y,group,bs='fs',k=8,xt={{'bs':'sf','bnd':{renamed_boundary!r},'nmax':30}},by=L)",
        knots={"X": knots["x"], "Y": knots["y"]},
        smoothing_params=[.3, .3], optimize_smoothing=False,
    ).fit(data=matrix)
    return {"known_boundary_tensor": (tensor, frame), "matrix_factor": (grouped, matrix)}


if __name__ == "__main__":
    for label, (model, data) in fit_examples().items():
        prediction, se = model.predict(data, return_se=True)
        print(f"{label}: {len(prediction)} predictions, finite={bool(np.isfinite(prediction).all() and np.isfinite(se).all())}")
