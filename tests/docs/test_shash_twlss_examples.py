"""Execute the two general-family examples through their public adapters."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("family,columns", [("shash", 4), ("twlss", 3)])
def test_general_family_example(family, columns):
    path = Path(__file__).parents[2] / "examples" / f"example_{family}_gamlss.py"
    example = runpy.run_path(str(path))
    model, data = example["fit_example"]()
    parameters = model.predict_parameters(data)
    assert parameters.shape == (len(data), columns)
    assert np.isfinite(parameters).all()
    assert np.isfinite(model.predict_point(data)).all()
    model.predict_components(data).validate_additive_reconstruction()
