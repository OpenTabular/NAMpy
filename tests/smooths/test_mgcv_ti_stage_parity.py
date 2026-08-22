from __future__ import annotations

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.linalg import column_space_projector, symmetric_spectrum
from nampy.gam.penalties import tensor_product_penalties
from nampy.gam.smooths.tensor.marginals import (
    build_tensor_product_components,
    tensor_marginal_fit_matrices,
)
from nampy.gam.smooths.tensor.ti import InteractionTensorProductSplineTerm
from nampy.gam.smooths.univariate.tp import ThinPlateSplineTerm
from tests.families.test_general_family_mgcv_parity import _general_newdata
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _make_gaussian_data,
    _normalize_python_formula_text,
    _run_mgcv_natparam_type3,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_predict_matrix,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression, pytest.mark.smooth_ti]

_TI_RAW_CASES = [
    pytest.param("ti_2d_cs_cs", id="ti_2d_cs_cs"),
    pytest.param("ti_2d_cs_ps", id="ti_2d_cs_ps"),
    pytest.param("ti_2d_ps_cs", id="ti_2d_ps_cs"),
]
_TI_PREDICTION_STAGE_CASES = [
    pytest.param("ti_2d_cs_cs", 5e-13, id="ti_2d_cs_cs"),
    pytest.param("ti_2d_cs_ps", 5e-12, id="ti_2d_cs_ps"),
    pytest.param("ti_2d_ps_cs", 5e-12, id="ti_2d_ps_cs"),
]
_TI_PREDICTION_STAGE_FORMULAS = {
    "ti_2d_cs_cs": 'ti(x0, x1, bs=["cs", "cs"], k=[5, 6])',
    "ti_2d_cs_ps": 'ti(x0, x1, bs=["cs", "ps"], k=[5, 6])',
    "ti_2d_ps_cs": 'ti(x0, x1, bs=["ps", "cs"], k=[5, 6])',
}


def _stage_tensor_data():
    return _make_gaussian_data(seed=220, n=120)


def _stage_tensor_by_data():
    data = _make_gaussian_data(seed=221, n=120)
    z = 0.8 + 0.25 * np.cos(np.asarray(data["x0"], dtype=np.float64))
    return data.assign(z=np.asarray(z, dtype=np.float64))


def _mixed_ti_formula(case_id, *, fixed):
    smooth = _TI_PREDICTION_STAGE_FORMULAS[case_id]
    if fixed:
        smooth = smooth[:-1] + ", sp=[0.7, 1.1])"
    return f"y ~ {smooth}"


def _mixed_ti_newdata(data):
    newdata = data.iloc[3::17].drop(columns=["y"]).copy()
    newdata.loc[newdata.index[0], "x0"] = float(data["x0"].min()) - 0.35
    newdata.loc[newdata.index[-1], "x1"] = float(data["x1"].max()) + 0.35
    return newdata


def _assert_prediction_covariance_matches_mgcv(
    gam,
    expected_snapshot,
    data,
    newdata,
    formula,
    *,
    unconditional,
    atol,
):
    """Compare Vp/Vc after mapping it into the identified prediction space."""
    actual_lpmatrix = np.asarray(gam.lpmatrix(newdata), dtype=np.float64)
    actual_vcov = np.asarray(
        gam.vcov(unconditional=unconditional), dtype=np.float64
    )
    actual = actual_lpmatrix @ actual_vcov @ actual_lpmatrix.T

    expected_lpmatrix = np.asarray(
        _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family="gaussian",
            method="REML" if unconditional else "fixed",
            type="lpmatrix",
            optimizer="newton" if unconditional else None,
            allow_live_run=True,
        )["pred"],
        dtype=np.float64,
    )
    covariance_key = "cov_unconditional" if unconditional else "cov_bayes"
    expected_vcov = np.asarray(
        expected_snapshot["fit"][covariance_key], dtype=np.float64
    )
    expected = expected_lpmatrix @ expected_vcov @ expected_lpmatrix.T

    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)


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


def _ti_raw_constructor_penalties(term, X):
    use_centered = list(term._marginal_is_centered)
    _, marginal_penalties, _, basis_dims, _, _ = build_tensor_product_components(
        term._marginals,
        X,
        use_centered=use_centered,
        apply_np=True,
    )
    return tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)


def _assert_ti_penalty_spectrum_invariant(
    actual_penalty,
    expected_penalty,
    *,
    shrinkage_floor_multiplicity=0,
):
    actual_spectrum = symmetric_spectrum(
        np.asarray(actual_penalty, dtype=np.float64)
    )
    expected_spectrum = symmetric_spectrum(
        np.asarray(expected_penalty, dtype=np.float64)
    )
    assert actual_spectrum.shape == expected_spectrum.shape

    floor_size = int(shrinkage_floor_multiplicity)
    if floor_size:
        # smooth.construct.cr.smooth.spec() assigns unequal shrinkage values
        # to an otherwise unidentified two-dimensional null eigenspace. After
        # ti() centers that marginal, the one retained floor eigenvalue depends
        # on the LAPACK-specific null-space rotation. Its multiplicity after
        # tensor.prod.penalties() is identified, but its exact value is not.
        for spectrum in (actual_spectrum, expected_spectrum):
            floor = spectrum[:floor_size]
            assert np.all(floor > 0.0)
            np.testing.assert_allclose(
                floor,
                np.full(floor_size, np.mean(floor), dtype=np.float64),
                atol=1e-10,
                rtol=1e-10,
            )
            dominant_floor = 0.1 * float(spectrum[floor_size])
            assert float(np.max(floor)) <= dominant_floor + 1e-12

    np.testing.assert_allclose(
        actual_spectrum[floor_size:],
        expected_spectrum[floor_size:],
        atol=1e-10,
        rtol=1e-10,
    )


def _ti_prediction_parameterization(data, *, by=None):
    term = InteractionTensorProductSplineTerm(
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


def _ti_stage_case_prediction(case_id):
    data = _stage_tensor_data()
    if case_id == "ti_2d_cs_cs":
        term = InteractionTensorProductSplineTerm(
            feature=["x0", "x1"],
            k=[5, 6],
            basis=["cs", "cs"],
        )
    elif case_id == "ti_2d_cs_ps":
        term = InteractionTensorProductSplineTerm(
            feature=["x0", "x1"],
            k=[5, 6],
            basis=["cs", "ps"],
        )
    elif case_id == "ti_2d_ps_cs":
        term = InteractionTensorProductSplineTerm(
            feature=["x0", "x1"],
            k=[5, 6],
            basis=["ps", "cs"],
        )
    else:
        raise AssertionError(f"Unhandled ti stage case {case_id!r}")

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
        _TI_PREDICTION_STAGE_FORMULAS[case_id],
        absorb_cons=True,
        scale_penalty=True,
    )
    return term, actual, np.asarray(expected["X"], dtype=np.float64)


def test_ti_tp_raw_constructor_invariants_match_mgcv():
    """Verify that ti tp raw constructor invariants match mgcv."""
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


def test_ti_prediction_parameterization_matches_mgcv_and_preserves_penalty_order():
    """
    Verify that ti prediction parameterization matches mgcv and preserves penalty order.
    """
    data = _stage_tensor_data()
    term, actual, new_xy = _ti_prediction_parameterization(data)
    expected_predict = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1"]],
        new_xy,
        'ti(x0, x1, bs=["tp", "cr"], k=[6, 6])',
        absorb_cons=True,
        scale_penalty=True,
    )
    expected_raw = _run_mgcv_raw_constructor(
        data[["x0", "x1"]],
        'ti(x0, x1, bs=["tp", "cr"], k=[6, 6])',
    )

    assert term._marginal_is_centered == [True, True]
    assert len(term.penalties) == len(expected_raw["S"]) == 2
    assert [np.linalg.matrix_rank(np.asarray(S, dtype=np.float64)) for S in term.penalties] == [
        np.linalg.matrix_rank(np.asarray(S, dtype=np.float64))
        for S in expected_raw["S"]
    ]
    np.testing.assert_allclose(
        actual,
        np.asarray(expected_predict["X"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_ti_numeric_by_prediction_parameterization_matches_mgcv_and_keeps_mgcv_marginal_centering():
    """
    Verify that ti numeric by prediction parameterization matches mgcv and keeps
    the upstream marginal centering.
    """
    data = _stage_tensor_by_data()
    term, actual, new_xyz = _ti_prediction_parameterization(data, by="z")
    expected_predict = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1", "z"]],
        new_xyz,
        'ti(x0, x1, by=z, bs=["tp", "cr"], k=[6, 6])',
        absorb_cons=True,
        scale_penalty=True,
    )

    assert term._by_state is not None
    assert term._by_state.name == "z"
    assert term._by_state.is_constant is False
    assert term._marginal_is_centered == [True, True]
    np.testing.assert_allclose(
        actual,
        np.asarray(expected_predict["X"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(("case_id", "atol"), _TI_PREDICTION_STAGE_CASES)
def test_ti_mixed_basis_prediction_parameterizations_match_mgcv(case_id, atol):
    """Verify that ti mixed basis prediction parameterizations match mgcv."""
    term, actual, expected = _ti_stage_case_prediction(case_id)

    assert len(term.penalties) == 2
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("case_id", _TI_RAW_CASES)
def test_ti_shrinkage_penalty_invariants_match_mgcv_raw_constructor(case_id):
    """Verify identifiable ti shrinkage spectra match mgcv raw construction."""
    data = _stage_tensor_data()
    formula = _TI_PREDICTION_STAGE_FORMULAS[case_id]
    basis = formula.split('bs=["', 1)[1].split('"]', 1)[0].split('", "')
    term = InteractionTensorProductSplineTerm(
        feature=["x0", "x1"],
        k=[5, 6],
        basis=basis,
    )
    X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    term.fit(X, ["x0", "x1"])
    expected = _run_mgcv_raw_constructor(
        data[["x0", "x1"]],
        formula,
    )

    actual_penalties = _ti_raw_constructor_penalties(term, X)
    assert len(actual_penalties) == len(expected["S"]) == 2
    for axis, (actual_penalty, expected_penalty) in enumerate(
        zip(actual_penalties, expected["S"], strict=True)
    ):
        floor_multiplicity = (
            int(actual_penalty.shape[0] // term._basis_dims[axis])
            if basis[axis] == "cs"
            else 0
        )
        _assert_ti_penalty_spectrum_invariant(
            actual_penalty,
            expected_penalty,
            shrinkage_floor_multiplicity=floor_multiplicity,
        )


@pytest.mark.parametrize("case_id", _TI_RAW_CASES)
def test_ti_mixed_shrinkage_fixed_sp_fit_and_prediction_covariance_match_mgcv(
    case_id,
):
    """Mixed cs ti terms retain fixed-SP fit, SE, EDF, and Vp behavior."""
    data = _stage_tensor_data()
    formula = _mixed_ti_formula(case_id, fixed=True)
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    actual_snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    expected_snapshot = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(formula),
        "gaussian",
        "fixed",
        allow_live_run=True,
    )

    np.testing.assert_allclose(
        actual_snapshot["predictions"]["response"],
        expected_snapshot["predictions"]["response"],
        atol=2e-8,
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        actual_snapshot["fit"]["edf_total"],
        expected_snapshot["fit"]["edf_total"],
        atol=2e-8,
        rtol=2e-8,
    )

    newdata = _mixed_ti_newdata(data)
    actual_prediction, actual_se = gam.predict(
        newdata, type="response", return_se=True
    )
    expected_prediction = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual_prediction,
        np.asarray(expected_prediction["pred"]).ravel(),
        atol=2e-8,
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected_prediction["se"]).ravel(),
        atol=2e-8,
        rtol=2e-8,
    )
    _assert_prediction_covariance_matches_mgcv(
        gam,
        expected_snapshot,
        data,
        newdata,
        formula,
        unconditional=False,
        atol=2e-8,
    )


@pytest.mark.parametrize("case_id", _TI_RAW_CASES)
def test_ti_mixed_shrinkage_optimized_reml_fit_and_inference_match_mgcv(case_id):
    """Mixed cs ti terms retain REML, SP, Vc, EDF, and term-SE behavior."""
    data = _stage_tensor_data()
    formula = _mixed_ti_formula(case_id, fixed=False)
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual_snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    expected_snapshot = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(formula),
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual_snapshot,
        expected_snapshot,
        pred_atol=3e-5,
        pred_rtol=3e-5,
        sp_log_atol=3e-3,
        criterion_atol=3e-3,
    )

    newdata = _mixed_ti_newdata(data)
    actual_terms, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    expected_terms = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="terms",
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual_terms, expected_terms["pred"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        actual_se, expected_terms["se"], atol=3e-5, rtol=3e-5
    )
    _assert_prediction_covariance_matches_mgcv(
        gam,
        expected_snapshot,
        data,
        newdata,
        formula,
        unconditional=True,
        atol=4e-5,
    )
