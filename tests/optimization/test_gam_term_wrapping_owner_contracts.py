from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _run_mgcv_gam_setup_assembly,
    _run_mgcv_predict_on_newdata,
)
from tests.parity.test_mgcv_output_parity import (
    _make_factor_by_data,
    _make_numeric_by_data,
    _make_transformed_formula_data,
)

pytestmark = [pytest.mark.surface_regression]


def _make_tensor_by_data(seed=211, n=160):
    data = _make_gaussian_data(seed=seed, n=n)
    z = 0.8 + 0.25 * np.cos(np.asarray(data["x0"], dtype=np.float64))
    return data.assign(z=np.asarray(z, dtype=np.float64))


def _build_from_formula(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def _fit_fixed_model(data: pd.DataFrame, formula, family="gaussian") -> GAM:
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    return gam


def _expected_predictor_block(gam: GAM, expected_lpmatrix: np.ndarray, index: int = 0):
    predictor = gam.compiled_model_.predictors[index]
    sl = gam.compiled_model_.predictor_full_slices[index]
    block = np.asarray(expected_lpmatrix[:, sl], dtype=np.float64)
    if getattr(predictor, "prediction_has_intercept", predictor.has_intercept):
        np.testing.assert_allclose(
            block[:, :1],
            np.ones((block.shape[0], 1), dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
        block = block[:, 1:]
    return block


def _assert_offset_payload_equal(actual, expected):
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        actual_list = [] if actual is None else list(actual)
        expected_list = [] if expected is None else list(expected)
        assert len(actual_list) == len(expected_list)
        for got, want in zip(actual_list, expected_list):
            if got is None or want is None:
                assert got is None and want is None
                continue
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float64),
                np.asarray(want, dtype=np.float64),
                atol=1e-12,
                rtol=0.0,
            )
        return

    if actual is None or expected is None:
        assert actual is None and expected is None
        return

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_numeric_by_wrapped_predictor_block_matches_mgcv_lpmatrix_slice():
    """
    Owner-contract coverage verifying that numeric by wrapped predictor block matches
    mgcv lpmatrix slice.
    """
    data = _make_numeric_by_data(seed=101, n=160)
    formula = 'y ~ s(x, by=z, bs="cr", k=8, sp=1.0)'

    gam = _fit_fixed_model(data, formula)
    expected = _run_mgcv_predict_on_newdata(
        data,
        data,
        formula,
        family="gaussian",
        method="fixed",
        type="lpmatrix",
        return_se=False,
    )

    predictor = gam.compiled_model_.predictors[0]
    actual = np.asarray(predictor.build_new_matrix(gam.X_), dtype=np.float64)
    expected_block = _expected_predictor_block(gam, np.asarray(expected["pred"]))

    assert len(predictor.compiled_terms) == 1
    assert predictor.compiled_terms[0].by_variable_info.name == "z"
    np.testing.assert_allclose(actual, expected_block, atol=1e-12, rtol=0.0)


def test_factor_by_wrapped_predictor_preserves_level_ownership_and_matches_mgcv():
    """
    Owner-contract coverage verifying that factor by wrapped predictor preserves level
    ownership and matches mgcv.
    """
    data = _make_factor_by_data(seed=107, n=180)
    formula = 'y ~ f + s(x, by=f, bs="cr", k=8, sp=1.0)'

    gam = _fit_fixed_model(data, formula)
    expected = _run_mgcv_predict_on_newdata(
        data,
        data,
        formula,
        family="gaussian",
        method="fixed",
        type="lpmatrix",
        return_se=False,
    )

    predictor = gam.compiled_model_.predictors[0]
    actual = np.asarray(predictor.build_new_matrix(gam.X_), dtype=np.float64)
    expected_block = _expected_predictor_block(gam, np.asarray(expected["pred"]))

    parametric_terms = [
        term for term in predictor.compiled_terms if str(term.term_type) == "parametric"
    ]
    smooth_terms = [
        term for term in predictor.compiled_terms if str(term.term_type) == "smooth"
    ]

    assert len(parametric_terms) == 2
    assert {term.label for term in parametric_terms} == {"f[b]", "f[c]"}
    assert len(smooth_terms) == 3
    assert [
        dict(term.metadata or {}).get("factor_by", {}).get("level")
        for term in smooth_terms
    ] == ["a", "b", "c"]
    for term in smooth_terms:
        factor_by = dict(term.metadata or {}).get("factor_by", {})
        assert factor_by["source_by"] == "f"

    np.testing.assert_allclose(actual, expected_block, atol=1e-12, rtol=0.0)


def test_transformed_formula_offset_is_resolved_before_wrapping_and_matches_mgcv():
    """
    Owner-contract coverage verifying that transformed formula offset is resolved before
    wrapping and matches mgcv.
    """
    data = _make_transformed_formula_data(seed=531, n=120)
    formula = (
        'I(y**2) ~ I(x**2) + s(I(z**2), bs="cr", k=6, sp=0.9)'
        " + offset(log(o + 1))"
    )

    built = _build_from_formula(formula, data)
    expected_setup = _run_mgcv_gam_setup_assembly(
        data=data,
        formula=formula,
        family="gaussian",
        method="fixed",
        select=False,
    )
    gam = _fit_fixed_model(data, formula)
    expected = _run_mgcv_predict_on_newdata(
        data,
        data,
        formula,
        family="gaussian",
        method="fixed",
        type="lpmatrix",
        return_se=False,
    )

    predictor = gam.compiled_model_.predictors[0]
    actual = np.asarray(predictor.build_new_matrix(gam.X_), dtype=np.float64)
    expected_block = _expected_predictor_block(gam, np.asarray(expected["pred"]))

    assert built.predictor_specs[0].offset_name == gam.formula_offset_name_
    assert built.predictor_specs[0].offset_name in built.working_data.columns
    _assert_offset_payload_equal(built.offsets, expected_setup["offset"])
    _assert_offset_payload_equal(gam.offset_train_, expected_setup["offset"])
    np.testing.assert_allclose(actual, expected_block, atol=1e-12, rtol=0.0)


def test_tensor_numeric_by_wrapped_predictor_block_matches_mgcv_lpmatrix_slice():
    """
    Owner-contract coverage verifying that tensor numeric by wrapped predictor block
    matches mgcv lpmatrix slice.
    """
    data = _make_tensor_by_data()
    formula = 'y ~ te(x0, x1, by=z, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])'

    gam = _fit_fixed_model(data, formula)
    expected = _run_mgcv_predict_on_newdata(
        data,
        data,
        formula,
        family="gaussian",
        method="fixed",
        type="lpmatrix",
        return_se=False,
    )

    predictor = gam.compiled_model_.predictors[0]
    actual = np.asarray(predictor.build_new_matrix(gam.X_), dtype=np.float64)
    expected_block = _expected_predictor_block(gam, np.asarray(expected["pred"]))

    assert len(predictor.compiled_terms) == 1
    assert predictor.compiled_terms[0].by_variable_info.name == "z"
    np.testing.assert_allclose(actual, expected_block, atol=1e-12, rtol=0.0)

