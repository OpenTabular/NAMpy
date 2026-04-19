"""Focused parity checks for ``gam_vcomp(rescale=True)``."""

from __future__ import annotations

import numpy as np
import pytest

from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _make_gaussian_data,
    _run_mgcv_gam_vcomp,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def test_gam_vcomp_rescale_true_matches_mgcv_gcv():
    data = _make_gaussian_data(seed=41, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        "gaussian",
        "GCV",
        rescale=True,
    )
    gam = _fit_nampy_model(data, formula, "gaussian", "GCV")

    actual = gam.gam_vcomp(rescale=True)

    np.testing.assert_allclose(
        _as_float_array(actual["vc"]),
        _as_float_array(expected["vc"]),
        atol=5e-8,
        rtol=0.0,
    )


def test_gam_vcomp_rescale_true_matches_mgcv_reml_ci():
    data = _make_gaussian_data(seed=42, n=140)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        "gaussian",
        "REML",
        rescale=True,
    )
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")

    actual = gam.gam_vcomp(rescale=True)

    expected_vc = _as_float_array(expected["vc"])
    np.testing.assert_allclose(
        _as_float_array(actual["vc"]),
        expected_vc[:1],
        atol=1e-4,
        rtol=0.0,
    )
