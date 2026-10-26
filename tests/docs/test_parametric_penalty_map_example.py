"""Mapped parametric penalties retain public prediction and persistence."""

import numpy as np

from examples.example_parametric_penalty_map import fit_example
from nampy.models import GAMRegressor


def test_parametric_penalty_map_example(tmp_path):
    model, frame, _, sp = fit_example()
    np.testing.assert_array_equal(model.gam_.smoothing_params, sp)
    components = model.predict_components(frame)
    components.validate_additive_reconstruction()
    assert np.isfinite(model.standard_errors(frame)).all()
    restored = GAMRegressor.load_model(model.save_model(tmp_path / "penalty-map.nampy"))
    np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
