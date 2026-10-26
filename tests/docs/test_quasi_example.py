"""Executable public quasi-family workflow, requiring no R runtime."""

import runpy
from pathlib import Path

import numpy as np


def test_quasi_example_mean_uncertainty_and_unavailable_likelihood():
    namespace = runpy.run_path(str(Path(__file__).parents[2] / "examples/example_quasi_gam.py"))
    models = namespace["fit_examples"]()
    assert [m.family.name for m in models] == ["quasipoisson", "quasibinomial", "quasi"]
    for model in models:
        fit, se = model.predict(return_se=True)
        assert fit.shape == se.shape == (120,)
        assert np.isfinite(fit).all() and np.isfinite(se).all()
        assert np.all(se >= 0)
        assert np.isnan(model.loglik()) and np.isnan(model.aic()) and np.isnan(model.bic())
    assert np.all((models[1].predict() > 0) & (models[1].predict() < 1))
