"""Separate persisted-state invariants from prediction arithmetic diagnostics."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy import GAMLSS
from nampy.gam.families.gamlss.twlss import TwlssFamily
from nampy.gam.model_state import _coef_full


@pytest.mark.parametrize("layout", ["C", "F", "unaligned"])
def test_twlss_natural_phi_matches_scalar_r_exp_for_each_layout(layout):
    """R's exp(rho) is scalar Mathlib even when eta has a strided column."""
    rho = np.array([-745.0, -2.0, -1.2, -1.1, -0.7, -0.1, 0.0, 0.03, 0.7, 1.0, 709.0])
    if layout == "unaligned":
        storage = np.empty(rho.size * 3 * 8 + 1, dtype=np.uint8)
        eta = np.ndarray((rho.size, 3), dtype=np.float64, buffer=storage, offset=1)
        assert not eta.flags.aligned
    else:
        eta = np.empty((rho.size, 3), order=layout)
        assert eta.flags.c_contiguous if layout == "C" else eta.flags.f_contiguous
    eta[:, :2] = 0.0
    eta[:, 2] = rho
    expected = np.array([math.exp(float(value)) for value in rho])
    np.testing.assert_array_equal(
        TwlssFamily().distribution_parameters_from_eta(eta)[:, 2], expected
    )


@pytest.fixture(scope="module")
def fitted_twlss():
    source = Path(__file__).parents[1] / "reference_fixtures/mgcv_twlss/inputs.json"
    data = pd.DataFrame(json.loads(source.read_text()))
    model = GAMLSS(
        formula=["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"],
        family=TwlssFamily(),
        smoothing_params=[0.7, 0.9],
        optimize_smoothing=False,
    ).fit(data)
    return model, data


def test_twlss_persistence_preserves_exact_state_and_linear_prediction(
    fitted_twlss, tmp_path
):
    """Report coefficient/design loss before any nonlinear output difference."""
    model, data = fitted_twlss
    restored = GAMLSS.load_model(model.save_model(tmp_path / "twlss.nampy"))
    np.testing.assert_array_equal(_coef_full(restored.gam_), _coef_full(model.gam_))
    np.testing.assert_array_equal(restored.gam_.vcov(), model.gam_.vcov())
    np.testing.assert_array_equal(
        restored.gam_.lpmatrix(data), model.gam_.lpmatrix(data)
    )
    np.testing.assert_array_equal(
        restored.predict(data, raw=True), model.predict(data, raw=True)
    )
    assert restored.gam_.family.a == model.gam_.family.a
    assert restored.gam_.family.b == model.gam_.family.b
    assert restored.gam_.family.link_names == model.gam_.family.link_names
