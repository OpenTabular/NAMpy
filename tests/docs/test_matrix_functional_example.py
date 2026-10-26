"""The documented matrix-functional example runs without R."""

import runpy
from pathlib import Path

import numpy as np


def test_matrix_functional_example_predictions_and_uncertainty():
    example = Path(__file__).parents[2] / "examples" / "example_matrix_functionals.py"
    data, surface, tensor = runpy.run_path(str(example))["fit_examples"]()
    for model in (surface, tensor):
        fit, se = model.predict(data, return_se=True)
        assert fit.shape == se.shape == (90,)
        assert np.isfinite(fit).all() and np.isfinite(se).all()
        assert np.all(se >= 0)
