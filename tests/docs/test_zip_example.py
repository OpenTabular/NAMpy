"""Executable public hurdle Poisson example; no R in normal runs."""

import runpy
from pathlib import Path

import numpy as np


def test_zip_example_fixed_distribution_parameters():
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_zip_gam.py")
    )
    model = namespace["fit_example"]()
    mean, se = model.predict(type="response", return_se=True)
    latent = model.predict(type="link")
    assert mean.shape == se.shape == latent.shape == (180,)
    assert np.isfinite(mean).all() and np.isfinite(se).all()
    assert np.all(mean >= 0) and np.all(se >= 0)
    np.testing.assert_allclose(mean, model.family.response_from_eta(latent))
    assert not np.allclose(mean, latent)
    np.testing.assert_allclose(model.family.getTheta(True), [-0.5, 0.8])
    assert model.family.n_theta == 0
    assert np.isfinite(model.loglik()) and np.isfinite(model.aic())
