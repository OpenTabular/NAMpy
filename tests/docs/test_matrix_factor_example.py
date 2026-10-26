"""Explicit matrix factors preserve the public additive and persistence contracts."""

import numpy as np

from examples.example_matrix_factor import fit_example
from nampy.models import GAMRegressor


def test_matrix_factor_example(tmp_path):
    model, frame = fit_example()
    model.predict_components(frame).validate_additive_reconstruction()
    prediction = model.predict(frame)
    assert np.isfinite(prediction).all()
    assert np.isfinite(model.standard_errors(frame)).all()
    restored = GAMRegressor.load_model(model.save_model(tmp_path / "matrix-factor.nampy"))
    np.testing.assert_array_equal(restored.predict(frame), prediction)
