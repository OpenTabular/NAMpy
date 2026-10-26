"""The censored-Poisson user example follows public additive contracts."""

import runpy
from pathlib import Path

import numpy as np


def test_cpois_example_executes_and_reconstructs():
    example = runpy.run_path(str(Path(__file__).parents[2] / "examples/example_cpois_gam.py"))
    model, data = example["fit_example"]()
    assert np.all(model.predict(data) > 0)
    model.predict_components(data).validate_additive_reconstruction()
    assert model.gam_._optim_result.success
