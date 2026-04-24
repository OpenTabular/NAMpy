from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.linalg import column_space_projector, symmetric_spectrum
from nampy.gam.smooths.algebra import t2_marginal_reparameterization
from nampy.gam.smooths.tensor.marginals import tensor_marginal_fit_matrices
from nampy.gam.smooths.tensor.t2 import TensorANOVASplineTerm
from nampy.gam.smooths.univariate.tp import ThinPlateSplineTerm
from tests.families.test_general_family_mgcv_parity import (
    _general_newdata,
    _gevlss_tensor_data,
    _shashlss_tensor_data,
    _ziplss_tensor_data,
)
from tests.mgcv_parity_utils import (
    _run_mgcv_natparam_type3,
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_predict_matrix,
)

pytestmark = [pytest.mark.surface_regression]

_TP_STAGE_CASES = [
    pytest.param(_gevlss_tensor_data, id="gevlss"),
    pytest.param(_shashlss_tensor_data, id="shashlss"),
    pytest.param(_ziplss_tensor_data, id="ziplss"),
]

_TP_END_TO_END_CASES = [
    pytest.param(_gevlss_tensor_data, id="gevlss"),
    pytest.param(_shashlss_tensor_data, id="shashlss"),
    pytest.param(
        _ziplss_tensor_data,
        id="ziplss",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "tp raw constructor exact orientation still differs on ziplss "
                    "tensor data; stage-local guard for downstream t2 lpmatrix gap."
                ),
            ),
        ],
    ),
]

_T2_PREDICT_CASES = [
    pytest.param("gevlss_t2_full_true", _gevlss_tensor_data, True, id="gevlss_t2_full_true"),
    pytest.param("shashlss_t2_full_true", _shashlss_tensor_data, True, id="shashlss_t2_full_true"),
    pytest.param(
        "ziplss_t2_full_false",
        _ziplss_tensor_data,
        False,
        id="ziplss_t2_full_false",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "t2(tp,cr) prediction parameterization still mismatches "
                    "mgcv PredictMat on ziplss full=FALSE tensor surface."
                ),
            ),
        ],
    ),
    pytest.param(
        "ziplss_t2_full_true",
        _ziplss_tensor_data,
        True,
        id="ziplss_t2_full_true",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "t2(tp,cr) prediction parameterization still mismatches "
                    "mgcv PredictMat on ziplss full=TRUE tensor surface."
                ),
            ),
        ],
    ),
]


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


def _runtime_t2_prediction_parameterization(data, *, full: bool):
    term = TensorANOVASplineTerm(
        feature=["x0", "x1"],
        k=[6, 6],
        basis=["tp", "cr"],
        full=full,
    )
    X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    term.fit(X, ["x0", "x1"])

    newdata = _general_newdata(data)
    X_new = newdata[["x0", "x1"]].to_numpy(dtype=np.float64)
    actual = np.asarray(term.transform_new(X_new), dtype=np.float64)
    pred_basis_map = dict(term.metadata or {}).get("prediction_basis_map", None)
    if pred_basis_map is not None:
        actual = actual @ np.asarray(pred_basis_map, dtype=np.float64)
    return actual, newdata[["x0", "x1"]]


@pytest.mark.parametrize("data_factory", _TP_STAGE_CASES)
def test_tp_raw_constructor_invariants_match_mgcv_on_tensor_case_data(data_factory):
    """Verify that tp raw constructor invariants match mgcv on tensor case data."""
    data = data_factory()
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


@pytest.mark.parametrize("data_factory", _TP_STAGE_CASES)
def test_tp_natparam_type3_kernel_matches_mgcv_on_same_raw_inputs(data_factory):
    """Verify that tp natparam type3 kernel matches mgcv on same raw inputs."""
    data = data_factory()
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


@pytest.mark.parametrize("data_factory", _TP_END_TO_END_CASES)
def test_tp_natparam_type3_end_to_end_matches_mgcv_on_tensor_case_data(data_factory):
    """Verify that tp natparam type3 end to end matches mgcv on tensor case data."""
    data = data_factory()
    raw_X, raw_S = _fit_tp_raw_marginal(data)
    expected = _mgcv_tp_natparam(data)

    actual = t2_marginal_reparameterization(raw_X, raw_S, basis_name="tp")
    got_X = np.column_stack([actual["B_range"], actual["B_null"]])
    got_P = np.column_stack([actual["T_range"], actual["T_null"]])

    np.testing.assert_allclose(got_X, expected["X"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got_P, expected["P"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(("case_id", "data_factory", "full"), _T2_PREDICT_CASES)
def test_t2_prediction_parameterization_matches_mgcv_predictmat(
    case_id,
    data_factory,
    full,
):
    """Verify that t2 prediction parameterization matches mgcv predictmat."""
    del case_id
    data = data_factory()
    actual, new_xy = _runtime_t2_prediction_parameterization(data, full=full)
    expected = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1"]],
        new_xy,
        (
            't2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)'
            if full
            else 't2(x0, x1, bs=["tp", "cr"], k=[6, 6])'
        ),
        absorb_cons=True,
        scale_penalty=True,
    )

    np.testing.assert_allclose(
        actual,
        np.asarray(expected["X"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
