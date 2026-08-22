"""Exact constructor/prediction parity for identified univariate SCOP bases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam.splines.shape import (
    build_scop_univariate_setup,
    predict_scop_univariate,
)
from tests.scam.scam_reference_utils import run_scam_raw_constructor

_UPSTREAM_BASIS_CODE = {
    "mpiby": "mpiBy",
    "mpdby": "mpdBy",
    "mdcvby": "mdcvBy",
    "mdcxby": "mdcxBy",
    "micvby": "micvBy",
    "micxby": "micxBy",
    "cvby": "cvBy",
    "cxby": "cxBy",
    "cpop": "cpop",
}


@pytest.mark.parametrize(
    "basis_code",
    [
        "mpi",
        "mpd",
        "mdcv",
        "mdcx",
        "micv",
        "micx",
        "cv",
        "cx",
        "po",
        "dpo",
        "ipo",
        "miso",
        "mifo",
        "mpiby",
        "mpdby",
        "mdcvby",
        "mdcxby",
        "micvby",
        "micxby",
        "cvby",
        "cxby",
        "cpop",
    ],
)
def test_global_scop_raw_constructor_and_prediction_match_scam(basis_code):
    rng = np.random.default_rng(734)
    x = rng.uniform(-2.3, 3.7, size=67)
    data = pd.DataFrame({"x": x})
    new_x = np.array(
        [np.min(x) - 0.8, np.min(x), -0.4, 0.2, np.max(x), np.max(x) + 1.1]
    )
    new_data = pd.DataFrame({"x": new_x})
    upstream_code = _UPSTREAM_BASIS_CODE.get(basis_code, basis_code)
    expected = run_scam_raw_constructor(
        data, f"s(x, bs='{upstream_code}', k=8, m=2)", new_data=new_data
    )
    actual = build_scop_univariate_setup(
        x, basis_code=basis_code, bs_dim=8, spline_order=2
    )

    expected_class = (
        "cpopspline.smooth" if basis_code == "cpop" else f"{upstream_code}.smooth"
    )
    assert expected["class_name"] == expected_class
    np.testing.assert_allclose(actual.knots, expected["knots"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        actual.basis_train, expected["X"], rtol=0.0, atol=2e-14
    )
    expected_cmx = expected["cmX"]
    expected_center = (
        np.zeros(actual.n_coef, dtype=np.float64)
        if expected_cmx is None or np.asarray(expected_cmx).size == 0
        else expected_cmx
    )
    np.testing.assert_allclose(actual.center, expected_center, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(actual.sigma, expected["Sigma"], rtol=0.0, atol=0.0)
    if basis_code != "cpop":
        np.testing.assert_allclose(
            actual.difference_matrix, expected["P"][0], rtol=0.0, atol=0.0
        )
    np.testing.assert_allclose(actual.penalty, expected["S"][0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual.positive_mask, expected["p_ident"])
    if basis_code != "cpop":
        np.testing.assert_allclose(
            actual.derivative_basis_1, expected["Xdf1"], rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            actual.derivative_basis_2, expected["Xdf2"], rtol=0.0, atol=2e-14
        )
    assert actual.rank == expected["rank"]
    assert actual.null_space_dim == expected["null_space_dim"]
    assert expected["C"].shape == (0, actual.n_coef)
    np.testing.assert_allclose(
        predict_scop_univariate(new_x, actual),
        expected["prediction"],
        rtol=0.0,
        atol=3e-14,
    )


@pytest.mark.parametrize("basis_order,penalty_order", [(2, 1), (0, 0), (3, 2)])
def test_cpop_basis_and_penalty_orders_match_scam(
    basis_order, penalty_order
):
    rng = np.random.default_rng(1668)
    x = rng.uniform(-1.4, 3.1, size=71)
    new_x = np.array([-5.2, np.min(x), -0.1, np.max(x), 6.7])
    expected = run_scam_raw_constructor(
        pd.DataFrame({"x": x}),
        f"s(x, bs='cpop', k=9, m=c({basis_order}, {penalty_order}))",
        new_data=pd.DataFrame({"x": new_x}),
    )
    actual = build_scop_univariate_setup(
        x,
        basis_code="cpop",
        bs_dim=9,
        spline_order=basis_order,
        penalty_order=penalty_order,
    )

    np.testing.assert_allclose(actual.knots, expected["knots"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.basis_train, expected["X"], rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(actual.penalty, expected["S"][0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        predict_scop_univariate(new_x, actual),
        expected["prediction"],
        rtol=0.0,
        atol=3e-14,
    )


@pytest.mark.parametrize("basis_code", ["lmpi", "lipl"])
def test_local_scop_constructor_and_prediction_match_scam(basis_code):
    rng = np.random.default_rng(1882)
    x = rng.uniform(-2.0, 3.4, size=83)
    change_point = 0.45
    new_x = np.array([np.min(x) - 0.7, -1.0, change_point, 2.1, np.max(x) + 0.9])
    expected = run_scam_raw_constructor(
        pd.DataFrame({"x": x}),
        f"s(x, bs='{basis_code}', k=12, m=2, xt=list(xc={change_point}))",
        new_data=pd.DataFrame({"x": new_x}),
    )
    actual = build_scop_univariate_setup(
        x,
        basis_code=basis_code,
        bs_dim=12,
        spline_order=2,
        change_point=change_point,
    )

    np.testing.assert_allclose(actual.knots, expected["knots"], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(actual.basis_train, expected["X"], rtol=0.0, atol=3e-14)
    expected_center = np.asarray(expected["cmX"], dtype=np.float64)
    if basis_code == "lipl":
        expected_center = expected_center[1 : actual.n_coef + 1]
    np.testing.assert_allclose(actual.center, expected_center, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(actual.sigma, expected["Sigma"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.difference_matrix, expected["P"][0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.penalty, expected["S"][0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual.positive_mask, expected["p_ident"])
    if basis_code == "lmpi":
        np.testing.assert_allclose(
            actual.derivative_basis_1, expected["Xdf1"], rtol=0.0, atol=3e-14
        )
        np.testing.assert_allclose(
            actual.derivative_basis_2, expected["Xdf2"], rtol=0.0, atol=3e-14
        )
    assert actual.rank == expected["rank"]
    assert actual.null_space_dim == expected["null_space_dim"]
    np.testing.assert_allclose(
        predict_scop_univariate(new_x, actual),
        expected["prediction"],
        rtol=0.0,
        atol=4e-14,
    )
