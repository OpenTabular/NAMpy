"""Quasi GAMs: mean/variance modeling without a response likelihood.

Run with ``python examples/example_quasi_gam.py``. References are
stats::quasipoisson/quasibinomial/quasi and mgcv::fix.family.ls/gam.fit3.
"""

import numpy as np
import pandas as pd

from nampy.gam import GAM


def fit_examples():
    rng = np.random.default_rng(731)
    x = np.linspace(-1, 1, 120)
    exposure = 1.2 + 0.2 * np.cos(x)
    mean = exposure * np.exp(0.4 + 0.6 * np.sin(3 * x))
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mean))
    data = pd.DataFrame({"x": x, "y": counts, "log_exposure": np.log(exposure)})
    count_model = GAM(
        family="quasipoisson", formula="y ~ s(x,bs='cr',k=7) + offset(log_exposure)",
        optimize_smoothing=True, smoothing_method="reml",
    ).fit(data=data)

    trials = np.full(len(x), 10)
    probability = 1 / (1 + np.exp(-0.8 * np.sin(3 * x)))
    successes = rng.binomial(trials, probability)
    grouped = pd.DataFrame({"x": x, "success": successes, "failure": trials - successes})
    proportion_model = GAM(
        family="quasibinomial", formula="cbind(success,failure) ~ s(x,bs='cr',k=7)",
        optimize_smoothing=True, smoothing_method="reml",
    ).fit(data=grouped)

    positive = np.exp(0.2 + 0.3 * x + rng.normal(0, 0.2, len(x)))
    variance_model = GAM(
        family={"name": "quasi", "link": "log", "variance": "mu^2"},
        formula="y ~ s(x,bs='cr',k=7)",
        optimize_smoothing=True, smoothing_method="reml",
    ).fit(data=pd.DataFrame({"x": x, "y": positive}))
    return count_model, proportion_model, variance_model


if __name__ == "__main__":
    for model in fit_examples():
        fit, se = model.predict(return_se=True)
        print(model.family.name, "mean range:", (fit.min(), fit.max()), "mean SE:", se.mean())
        print("AIC is unavailable:", model.aic())
