"""Explicit B-spline constraints reach the shared public estimator contracts."""

import numpy as np

from examples.example_bs_constraints import fit_example


def test_bs_constraints_example():
    model, frame, constraint = fit_example()
    components = model.predict_components(frame)
    components.validate_additive_reconstruction()
    np.testing.assert_allclose(components.link[:, 0], model.predict(frame), atol=2e-12)
    assert np.isfinite(model.standard_errors(frame)).all()
    smooth = model.gam_.gam_result_.require_compiled_model().compiled_terms[0]
    np.testing.assert_allclose(
        constraint @ smooth.predict_coefficient_map, 0, atol=2e-12
    )
