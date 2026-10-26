"""The logistic example retains censor metadata and additive accounting."""

import runpy
from pathlib import Path

import numpy as np


def test_clog_example_predicts_location_and_retains_bounds():
    namespace = runpy.run_path(str(Path(__file__).parents[2] / "examples/example_clog_gam.py"))
    data = namespace["example_data"]()
    model = namespace["fit_example"]()
    components = model.predict_components(data[["x"]])
    components.validate_additive_reconstruction()
    assert components.response.shape == (160,)
    assert np.isfinite(components.response).all()
    np.testing.assert_array_equal(model.gam_.response_.bounds, data["bound"])
    np.testing.assert_allclose(model.gam_.family.getTheta(True), [0.7])
    estimated = namespace["fit_estimated_example"]()
    estimated.predict_components(data[["x"]]).validate_additive_reconstruction()
    assert estimated.gam_._optim_result.success
    assert estimated.gam_.family.getTheta(True)[0] > 0
