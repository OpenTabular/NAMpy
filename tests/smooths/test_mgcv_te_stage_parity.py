from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.linalg import column_space_projector, symmetric_spectrum
from nampy.gam.smooths.algebra import t2_marginal_reparameterization
from nampy.gam.smooths.tensor.marginals import tensor_marginal_fit_matrices
from nampy.gam.smooths.tensor.te import TensorProductSplineTerm
from nampy.gam.smooths.univariate.tp import ThinPlateSplineTerm
from tests.families.test_general_family_mgcv_parity import _general_newdata
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _run_mgcv_natparam_type3,
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_predict_matrix,
)
from tests.smooths.test_mgcv_raw_constructor_parity import (
    CASES as RAW_CONSTRUCTOR_CASES,
)

pytestmark = [pytest.mark.surface_regression, pytest.mark.smooth_te]

_RAW_CASES_BY_ID = {case.case_id: case for case in RAW_CONSTRUCTOR_CASES}
_TE_MIXED_BASIS_STAGE_CASES = [
    pytest.param("te_2d_cr_cs", 2e-3, id="te_2d_cr_cs"),
    pytest.param("te_2d_tp_ts", 5e-8, id="te_2d_tp_ts"),
]


def _stage_tensor_data():
    return _make_gaussian_data(seed=220, n=120)


def _stage_tensor_by_data():
    data = _make_gaussian_data(seed=221, n=120)
    z = 0.8 + 0.25 * np.cos(np.asarray(data["x0"], dtype=np.float64))
    return data.assign(z=np.asarray(z, dtype=np.float64))


def _fit_tp_raw_marginal(data):
    term = ThinPlateSplineTerm(feature="x0", k=6, basis="tp")
    X = data[["x0"]].to_numpy(dtype=np.float64)
    term.fit(X, ["x0"])
    B, S, _ = tensor_marginal_fit_matrices(term, centered=False)
    return np.asarray(B, dtype=np.float64), np.asarray(S, dtype=np.float64)


def _mgcv_tp_raw_constructor(data):
    expected = _run_mgcv_raw_constructor(data[["x0"]], 's(x0, bs="tp", k=6)')
    return (
        np.asarray(expected["X"], dtype=np.float64),
        np.asarray(expected["S"][0], dtype=np.float64),
    )


def _mgcv_tp_natparam(data):
    expected = _run_mgcv_natparam_type3(data[["x0"]], 's(x0, bs="tp", k=6)')
    return {
        "rawX": np.asarray(expected["rawX"], dtype=np.float64),
        "rawS": np.asarray(expected["rawS"], dtype=np.float64),
        "X": np.asarray(expected["X"], dtype=np.float64),
        "P": np.asarray(expected["P"], dtype=np.float64),
    }


def _te_prediction_parameterization(data, *, by=None):
    term = TensorProductSplineTerm(
        feature=["x0", "x1"],
        k=[6, 6],
        basis=["tp", "cr"],
        by=by,
    )
    fit_cols = ["x0", "x1"] + ([] if by is None else [by])
    X = data[fit_cols].to_numpy(dtype=np.float64)
    term.fit(X, fit_cols)

    newdata = _general_newdata(data)
    actual = np.asarray(
        term.transform_new(newdata[fit_cols].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    return term, actual, newdata[fit_cols]


def _te_stage_case_prediction(case_id):
    case = _RAW_CASES_BY_ID[case_id]
    data = case.data_factory()
    if case_id == "te_2d_cr_cs":
        term = TensorProductSplineTerm(
            feature=["x0", "x1"],
            k=[5, 6],
            basis=["cr", "cs"],
        )
    elif case_id == "te_2d_tp_ts":
        term = TensorProductSplineTerm(
            feature=["x0", "x1"],
            k=[5, 6],
            basis=["tp", "ts"],
        )
    else:
        raise AssertionError(f"Unhandled te stage case {case_id!r}")

    X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    term.fit(X, ["x0", "x1"])

    newdata = _general_newdata(data)
    actual = np.asarray(
        term.transform_new(newdata[["x0", "x1"]].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    expected = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1"]],
        newdata[["x0", "x1"]],
        case.formula.split("~", 1)[1].strip(),
        absorb_cons=True,
        scale_penalty=True,
    )
    return term, actual, np.asarray(expected["X"], dtype=np.float64)


def test_te_tp_raw_constructor_invariants_match_mgcv():
    """Verify that te tp raw constructor invariants match mgcv."""
    data = _stage_tensor_data()
    actual_X, actual_S = _fit_tp_raw_marginal(data)
    expected_X, expected_S = _mgcv_tp_raw_constructor(data)

    np.testing.assert_allclose(
        column_space_projector(actual_X),
        column_space_projector(expected_X),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        symmetric_spectrum(actual_S),
        symmetric_spectrum(expected_S),
        atol=1e-10,
        rtol=1e-10,
    )


def test_te_tp_natparam_type3_kernel_matches_mgcv_on_same_raw_inputs():
    """Verify that te tp natparam type3 kernel matches mgcv on same raw inputs."""
    data = _stage_tensor_data()
    expected = _mgcv_tp_natparam(data)

    actual = t2_marginal_reparameterization(
        expected["rawX"],
        expected["rawS"],
        basis_name="tp",
    )
    got_X = np.column_stack([actual["B_range"], actual["B_null"]])
    got_P = np.column_stack([actual["T_range"], actual["T_null"]])

    np.testing.assert_allclose(got_X, expected["X"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got_P, expected["P"], atol=1e-12, rtol=1e-12)


def test_te_prediction_parameterization_matches_mgcv_predict_matrix():
    """Verify that te prediction parameterization matches mgcv predict matrix."""
    data = _stage_tensor_data()
    _term, actual, new_xy = _te_prediction_parameterization(data)
    expected = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1"]],
        new_xy,
        'te(x0, x1, bs=["tp", "cr"], k=[6, 6])',
        absorb_cons=True,
        scale_penalty=True,
    )

    np.testing.assert_allclose(
        actual,
        np.asarray(expected["X"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_te_numeric_by_prediction_parameterization_matches_mgcv_predict_matrix():
    """
    Verify that te numeric by prediction parameterization matches mgcv predict matrix.
    """
    data = _stage_tensor_by_data()
    term, actual, new_xyz = _te_prediction_parameterization(data, by="z")
    expected = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1", "z"]],
        new_xyz,
        'te(x0, x1, by=z, bs=["tp", "cr"], k=[6, 6])',
        absorb_cons=True,
        scale_penalty=True,
    )

    assert term._by_state is not None
    assert term._by_state.name == "z"
    assert term._by_state.is_constant is False
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["X"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(("case_id", "atol"), _TE_MIXED_BASIS_STAGE_CASES)
def test_te_mixed_basis_prediction_parameterizations_match_mgcv(case_id, atol):
    """Verify that te mixed basis prediction parameterizations match mgcv."""
    term, actual, expected = _te_stage_case_prediction(case_id)

    assert len(term.penalties) == 2
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
