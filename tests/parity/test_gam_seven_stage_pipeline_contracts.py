"""Direct contracts for the seven public GAM pipeline stages in ``CLAUDE.md``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.state import FitCoreSolution
from nampy.gam.smooths.univariate.cr import CubicSplineTerm
from tests.mgcv_parity_utils import _run_mgcv_predict_on_newdata

pytestmark = [pytest.mark.surface_regression]

_FORMULA = 'y ~ z + s(x, bs="cr", k=6, sp=0.8) + offset(off)'


@dataclass(frozen=True)
class _PipelineCase:
    data: pd.DataFrame
    newdata: pd.DataFrame
    gam: GAM


@pytest.fixture(scope="module")
def pipeline_case() -> _PipelineCase:
    x = np.linspace(-1.25, 1.25, 96)
    z = 0.4 + np.cos(1.3 * x)
    off = 0.15 * np.sin(0.7 * x)
    y = off + 0.55 * z + np.sin(np.pi * x) + 0.04 * np.cos(7.0 * x)
    data = pd.DataFrame({"y": y, "x": x, "z": z, "off": off})

    new_x = np.linspace(-1.1, 1.1, 19)
    newdata = pd.DataFrame(
        {
            "x": new_x,
            "z": 0.4 + np.cos(1.3 * new_x),
            "off": 0.15 * np.sin(0.7 * new_x),
        }
    )
    gam = GAM(
        family="gaussian",
        formula=_FORMULA,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    return _PipelineCase(data=data, newdata=newdata, gam=gam)


def _smooth_term(case: _PipelineCase):
    return next(
        term
        for term in case.gam.compiled_model_.compiled_terms
        if term.term_type == "smooth"
    )


def test_stage_1_formula_and_specs_preserve_declarative_contract(pipeline_case):
    """Stage 1 owns parsing, offsets, and canonical predictor/term specs."""
    gam = pipeline_case.gam
    parsed = gam.formula_
    predictor = gam.predictor_specs[0]

    assert parsed.response_name == "y"
    assert parsed.nlp == 1
    assert predictor.name == "eta1"
    assert predictor.has_intercept is True
    assert predictor.offset_name == "off"
    assert [(term.kind, term.features) for term in predictor.terms] == [
        ("parametric", ("z",)),
        ("smooth", ("x",)),
    ]
    smooth_spec = predictor.terms[1].smooth_spec
    assert smooth_spec is not None
    assert smooth_spec.bs == "cr"
    assert smooth_spec.k == 6
    assert smooth_spec.sp == 0.8
    assert set(gam.formula_used_columns_) == {"x", "z", "off"}
    assert gam.formula_offset_names_ == ("off",)


def test_stage_2_runtime_term_owns_basis_semantics(pipeline_case):
    """Stage 2 keeps constructor state and prediction transforms on the runtime."""
    gam = pipeline_case.gam
    term = _smooth_term(pipeline_case)
    runtime = term.predict_fn.__self__

    assert isinstance(runtime, CubicSplineTerm)
    assert runtime.basis_name == "cr"
    assert runtime._feature_name == "x"
    assert runtime.basis_train.shape[0] == len(pipeline_case.data)
    runtime_prediction_basis = np.asarray(term.predict_fn(gam.X_), dtype=np.float64)
    np.testing.assert_allclose(
        runtime_prediction_basis,
        np.asarray(runtime.basis_train, dtype=np.float64),
        rtol=0.0,
        atol=1e-13,
    )
    penalty_defs = runtime.get_penalty_definitions()
    assert len(penalty_defs) == 1
    assert penalty_defs[0].matrix.shape[0] == runtime.basis_train.shape[1]


def test_stage_3_constructed_term_pairs_basis_penalty_and_prediction(pipeline_case):
    """Stage 3 keeps each wrapped basis paired with its penalty and predictor."""
    gam = pipeline_case.gam
    compiled = gam.compiled_model_
    term = _smooth_term(pipeline_case)
    term_index = compiled.compiled_terms.index(term)
    penalties = [
        penalty
        for penalty in compiled.compiled_penalties
        if penalty.term_index == term_index
    ]

    assert len(penalties) == 1
    penalty = penalties[0]
    width = term.coef_slice.stop - term.coef_slice.start
    assert penalty.coef_slice == term.coef_slice
    assert penalty.matrix.shape == (width, width)
    assert penalty.sp_mode == "fixed"
    assert penalty.sp_value == 0.8
    np.testing.assert_allclose(
        term.predict_matrix(gam.X_), term.basis_train, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        penalty.matrix,
        np.asarray(term.penalty_specs[0].matrix, dtype=np.float64),
        rtol=0.0,
        atol=1e-13,
    )


def test_stage_4_compiler_preserves_block_and_smoothing_order(pipeline_case):
    """Stage 4 assembles terms, slices, penalties, and smoothing indices once."""
    compiled = pipeline_case.gam.compiled_model_
    predictor = compiled.predictors[0]
    term_blocks = [
        np.asarray(term.basis_train, dtype=np.float64)
        for term in predictor.compiled_terms
    ]

    np.testing.assert_allclose(
        np.column_stack(term_blocks), predictor.design_matrix, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        predictor.design_matrix, compiled.design_matrix, rtol=0.0, atol=0.0
    )
    assert [term.coef_slice.start for term in predictor.compiled_terms] == [0, 1]
    assert predictor.compiled_terms[-1].coef_slice.stop == compiled.n_coef
    assert compiled.n_smoothing_params == 1
    assert [penalty.smoothing_index for penalty in compiled.compiled_penalties] == [0]
    np.testing.assert_allclose(pipeline_case.gam.smoothing_params, [0.8])


def test_stage_5_side_conditions_pair_fit_and_prediction_transforms(pipeline_case):
    """Stage 5 reports identifiability work and preserves train/predict pairing."""
    gam = pipeline_case.gam
    compiled = gam.compiled_model_
    predictor = compiled.predictors[0]
    reports = gam.side_condition_reports_

    assert reports is not None and len(reports) == 1
    assert reports[0]["predictor"] == "eta1"
    assert [item["label"] for item in reports[0]["term_reports"]] == [
        term.label for term in predictor.compiled_terms
    ]
    assert predictor.side_condition_Q is not None
    assert predictor.side_condition_Q.shape[1] == predictor.n_coef
    np.testing.assert_allclose(
        predictor.build_new_matrix(gam.X_),
        predictor.design_matrix,
        rtol=0.0,
        atol=1e-12,
    )
    expected_lpmatrix = np.column_stack(
        [np.ones(len(pipeline_case.data)), predictor.design_matrix]
    )
    np.testing.assert_allclose(
        gam.lpmatrix(pipeline_case.data),
        expected_lpmatrix,
        rtol=0.0,
        atol=1e-12,
    )


def test_stage_6_fit_solution_is_consistent_with_compiled_design(pipeline_case):
    """Stage 6 returns the stable fit contract in the public parameterization."""
    gam = pipeline_case.gam
    solution = gam.fit_core_solution_
    result = solution.fit_result
    lpmatrix = gam.lpmatrix(pipeline_case.data)
    expected_eta = (
        lpmatrix @ np.asarray(result.coef_full, dtype=np.float64)
        + pipeline_case.data["off"].to_numpy(dtype=np.float64)
    )

    assert isinstance(solution, FitCoreSolution)
    assert solution.penalized_system.X is not None
    assert solution.fit_state.X is not None
    np.testing.assert_allclose(
        solution.penalized_system.X, solution.fit_state.X, rtol=0.0, atol=0.0
    )
    assert result.coef_full.shape == (lpmatrix.shape[1],)
    np.testing.assert_allclose(result.eta, expected_eta, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(result.mu, expected_eta, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(
        solution.fit_state.X[:, 1:],
        gam.compiled_model_.design_matrix,
        rtol=0.0,
        atol=0.0,
    )
    fit_summary = gam.fit_result(include_covariances=True)
    np.testing.assert_allclose(fit_summary.coef_full, result.coef_full)
    np.testing.assert_allclose(fit_summary.edf_total, result.trace_H)


def test_stage_7_prediction_and_diagnostics_match_mgcv_and_fitted_state(
    pipeline_case,
):
    """Stage 7 consumes the fitted state consistently and matches mgcv newdata."""
    gam = pipeline_case.gam
    newdata = pipeline_case.newdata
    lpmatrix = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)
    link = np.asarray(gam.predict(newdata, type="link"), dtype=np.float64)
    response = np.asarray(gam.predict(newdata, type="response"), dtype=np.float64)
    terms = np.asarray(gam.predict(newdata, type="terms"), dtype=np.float64)
    coef = np.asarray(gam.fit_core_solution_.fit_result.coef_full, dtype=np.float64)
    offset = newdata["off"].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(link, lpmatrix @ coef + offset, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(response, link, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        link,
        float(coef[0]) + np.sum(terms, axis=1) + offset,
        rtol=0.0,
        atol=1e-12,
    )
    training_response = np.asarray(
        gam.predict(pipeline_case.data, type="response"), dtype=np.float64
    )
    np.testing.assert_allclose(
        gam.residuals(type="response"),
        pipeline_case.data["y"].to_numpy(dtype=np.float64) - training_response,
        rtol=0.0,
        atol=5e-16,
    )

    expected = _run_mgcv_predict_on_newdata(
        pipeline_case.data,
        newdata,
        _FORMULA,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=False,
    )
    np.testing.assert_allclose(
        response,
        np.asarray(expected["pred"], dtype=np.float64).ravel(),
        rtol=2e-7,
        atol=2e-7,
    )
