"""Normal and gamma GAMLSS fits with separate location and scale predictors.

Run with ``python examples/example_gamlss.py``.
"""

import numpy as np
import pandas as pd

from nampy import GAMLSS


def fit_normal():
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 1.0, 300)
    group = rng.choice(["a", "b"], size=x.size)
    mu = np.sin(2 * np.pi * x) + 0.25 * (group == "b")
    sigma = 0.08 + 0.30 * np.sin(2 * np.pi * x) ** 2
    data = pd.DataFrame(
        {
            "x": x,
            "group": group,
            "y": mu + rng.normal(scale=sigma),
        }
    )
    model = GAMLSS(
        family="normal",
        formula={
            "mu": "y ~ s(x, bs='cr', k=10) + group",
            "sigma": "~ s(x, bs='cr', k=8)",
        },
    ).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


def fit_gamma():
    rng = np.random.default_rng(19)
    x = np.linspace(0.0, 1.0, 240)
    mu = np.exp(0.4 + 0.8 * np.sin(2 * np.pi * x))
    sigma = 0.25 + 0.15 * x
    shape = 1.0 / sigma**2
    data = pd.DataFrame({"x": x, "y": rng.gamma(shape, mu / shape)})
    model = GAMLSS(
        family="gamma",
        formula={
            "mu": "y ~ s(x, bs='cr', k=8)",
            "sigma": "~ s(x, bs='cr', k=6)",
        },
    ).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    return model, data


def report(name, model, data):
    parameters = model.predict_parameters(data)
    point = model.predict_point(data)
    interval = model.predict_quantiles(data, [0.1, 0.9])
    print(name, "parameters:", model.parameter_names_)
    print("  point range:", (float(point.min()), float(point.max())))
    print("  parameter means:", parameters.mean(axis=0))
    print("  central 80% width:", float(np.mean(interval[:, 1] - interval[:, 0])))
    print("  mean log likelihood:", float(model.score(data, data["y"])))


if __name__ == "__main__":
    for name, fit in (("normal", fit_normal), ("gamma", fit_gamma)):
        fitted, frame = fit()
        report(name, fitted, frame)
