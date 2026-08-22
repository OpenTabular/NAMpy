"""Bivariate SCAM constructor and prediction parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam.penalties.algebra import scale_penalty
from nampy.gam.splines.shape import (
    build_bivariate_shape_setup,
    predict_bivariate_shape,
)
from tests.scam.scam_reference_utils import run_scam_raw_constructor


@pytest.mark.parametrize(
    "basis_code",
    [
        "tedmi",
        "tedmd",
        "temicx",
        "temicv",
        "tedecv",
        "tedecx",
        "tecvcv",
        "tecxcx",
        "tecxcv",
        "tescv",
        "tescx",
        "tesmi1",
        "tesmd1",
        "tesmi2",
        "tesmd2",
        "tismi",
        "tismd",
    ],
)
def test_double_monotone_bivariate_constructor_and_prediction_match_scam(
    basis_code,
):
    rng = np.random.default_rng(3101)
    data = pd.DataFrame(
        {
            "x": rng.uniform(-1.8, 2.4, size=83),
            "z": rng.uniform(-2.1, 1.7, size=83),
        }
    )
    new_data = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.6, 29),
            "z": np.linspace(1.9, -2.3, 29),
        }
    )
    expected = run_scam_raw_constructor(
        data,
        f"s(x, z, bs='{basis_code}', k=c(6, 7), m=c(2, 1))",
        new_data=new_data,
        smoothcon=True,
    )
    setup = build_bivariate_shape_setup(
        data["x"],
        data["z"],
        basis_code=basis_code,
        bs_dim=(6, 7),
        spline_order=(2, 1),
    )

    np.testing.assert_allclose(setup.basis_train, expected["X"], atol=5e-14)
    for actual, reference in zip(setup.penalties, expected["S"], strict=True):
        np.testing.assert_allclose(
            scale_penalty(setup.basis_train, actual), reference, atol=5e-14
        )
    np.testing.assert_array_equal(setup.positive_mask, expected["p_ident"])
    np.testing.assert_allclose(setup.knots[0], expected["knots"][0], atol=0.0)
    np.testing.assert_allclose(setup.knots[1], expected["knots"][1], atol=0.0)
    np.testing.assert_allclose(
        predict_bivariate_shape(new_data["x"], new_data["z"], setup),
        expected["prediction"],
        atol=8e-14,
    )
