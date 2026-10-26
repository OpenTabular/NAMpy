"""Executable public SCAT example; normal runs do not require R."""

import runpy
from pathlib import Path

import numpy as np


def test_scat_example_fixed_distribution_parameters():
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_scat_gam.py")
    )
    model = namespace["fit_example"]()
    mean, se = model.predict(return_se=True)
    assert mean.shape == se.shape == (100,)
    assert np.isfinite(mean).all() and np.isfinite(se).all()
    assert np.all(se >= 0)
    np.testing.assert_allclose(model.family.getTheta(True), [5, 0.3])
    assert model.family.n_theta == 0
    assert np.isfinite(model.loglik()) and np.isfinite(model.aic())
