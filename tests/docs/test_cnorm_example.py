"""Executable mixed-censor normal example; no R in normal runs."""

import runpy
from pathlib import Path

import numpy as np


def test_cnorm_example_preserves_censor_pairing_and_predicts_location():
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_cnorm_gam.py")
    )
    data = namespace["example_data"]()
    model = namespace["fit_example"]()
    value, bound = data["value"].to_numpy(), data["bound"].to_numpy()
    assert np.any(value == bound)
    assert np.any(np.isneginf(bound)) and np.any(np.isposinf(bound))
    assert np.any(np.isfinite(bound) & (value != bound))
    np.testing.assert_array_equal(model.response_.values, value)
    np.testing.assert_array_equal(model.response_.bounds, bound)
    assert not model.response_.values.flags.writeable
    location, se = model.predict(data[["x"]], type="response", return_se=True)
    assert location.shape == se.shape == (120,)
    assert np.isfinite(location).all() and np.isfinite(se).all()
    assert np.all(se >= 0)
    np.testing.assert_allclose(location, model.predict(type="link"))
    np.testing.assert_allclose(model.family.getTheta(True), [1])
    assert model.family.n_theta == 0
    assert np.isfinite(model.loglik()) and np.isfinite(model.aic())
