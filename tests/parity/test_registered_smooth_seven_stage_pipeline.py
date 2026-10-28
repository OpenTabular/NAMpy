"""Owner contracts for every registered smooth, beyond constructor-only coverage.

Seven stages are tested separately so failures identify the broken boundary.
No cross-library claim is made by internal reconstruction; the regular smooth
source comparisons and existing SCAM fixtures supply independent evidence.
"""

import numpy as np
import pytest

import nampy.gam.smooths  # noqa: F401
import nampy.gam.specs  # noqa: F401
from nampy import GAMRegressor
from nampy.gam.basis_registry import basis_descriptors
from nampy.gam.fit.state import FitCoreSolution
from tests.parity._seven_stage_cases import (
    MULTIVARIATE_ADAPTIVE_CASES,
    REGULAR_BASES,
    SHAPE_BASES,
    fit_smooth_case,
)

pytestmark = [pytest.mark.surface_regression]


@pytest.fixture(
    scope="module", params=REGULAR_BASES + SHAPE_BASES + MULTIVARIATE_ADAPTIVE_CASES
)
def smooth_pipeline(request):
    return fit_smooth_case(request.param)


def test_registry_has_explicit_seven_stage_cases():
    assert set(REGULAR_BASES).isdisjoint(SHAPE_BASES)
    assert set(REGULAR_BASES + SHAPE_BASES) == {d.name for d in basis_descriptors()}


def test_stage_1_formula_spec_and_offset(smooth_pipeline):
    case = smooth_pipeline
    model = case.model
    spec = model.predictor_specs[0]
    assert model.formula_.response_name == "y"
    assert model.formula_.nlp == 1
    assert spec.has_intercept and spec.offset_name == "off"
    assert len(spec.terms) == 1
    smooth = spec.terms[0].smooth_spec
    if case.basis in {"te", "ti", "t2"}:
        assert smooth.special == case.basis and smooth.bs == ["cr", "ps"]
    else:
        assert smooth.bs == case.basis
    np.testing.assert_array_equal(model.offset_train_, case.data.off)
    np.testing.assert_array_equal(model.prior_weights_, case.data.w)


def test_stage_2_runtime_basis_and_penalty_ownership(smooth_pipeline):
    case = smooth_pipeline
    term = case.compiled.compiled_terms[0]
    runtime = term.predict_fn.__self__
    assert runtime.basis_train.shape[0] == len(case.data)
    assert runtime.get_penalty_definitions()
    assert term.basis_name.split("(")[0] == case.basis
    # Runtime coordinates may precede compiler constraints, so compare its own
    # training/newdata pair rather than assuming its columns equal fitted ones.
    np.testing.assert_allclose(
        term.predict_fn(case.model.X_), runtime.basis_train, atol=1e-10, rtol=1e-10
    )
    for penalty in runtime.get_penalty_definitions():
        matrix = np.asarray(penalty.matrix)
        assert matrix.shape == (runtime.basis_train.shape[1],) * 2
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10)


def test_stage_3_constructed_term_retains_penalty_and_prediction(smooth_pipeline):
    case = smooth_pipeline
    for term in case.compiled.compiled_terms:
        np.testing.assert_allclose(
            term.predict_matrix(case.model.X_), term.basis_train, atol=2e-10, rtol=2e-10
        )
    for block in case.compiled.compiled_penalties:
        term = case.compiled.compiled_terms[block.term_index]
        width = term.coef_slice.stop - term.coef_slice.start
        assert block.coef_slice == term.coef_slice
        assert block.matrix.shape == (width, width)
        assert np.isfinite(block.matrix).all()
        np.testing.assert_allclose(block.matrix, block.matrix.T, atol=1e-10, rtol=1e-10)
        assert np.linalg.eigvalsh(block.matrix).min() >= -1e-8 * max(
            1, np.linalg.norm(block.matrix, 2)
        )


def test_stage_4_compiler_blocks_and_smoothing_indices(smooth_pipeline):
    case = smooth_pipeline
    compiled = case.compiled
    predictor = compiled.predictors[0]
    np.testing.assert_array_equal(
        np.column_stack([t.basis_train for t in predictor.compiled_terms]),
        predictor.design_matrix,
    )
    np.testing.assert_array_equal(compiled.design_matrix, predictor.design_matrix)
    assert compiled.n_smoothing_params == len(case.model.smoothing_params)
    assert {p.smoothing_index for p in compiled.compiled_penalties} == set(
        range(compiled.n_smoothing_params)
    )
    np.testing.assert_allclose(case.model.smoothing_params, 0.4, atol=0, rtol=0)
    assert sorted(np.concatenate(compiled.predictor_full_indices).tolist()) == list(
        range(len(case.model.fit_result().coef_full))
    )


def test_stage_5_identifiability_and_newdata_pairing(smooth_pipeline):
    case = smooth_pipeline
    compiled = case.compiled
    assert len(compiled.side_condition_reports) == 1
    predictor = compiled.predictors[0]
    assert compiled.side_condition_reports[0]["predictor"] == predictor.name
    np.testing.assert_allclose(
        predictor.build_new_matrix(case.model.X_),
        predictor.design_matrix,
        atol=2e-10,
        rtol=2e-10,
    )
    X = case.model.lpmatrix(case.data)
    np.testing.assert_allclose(X[:, 1:], compiled.design_matrix, atol=2e-10, rtol=2e-10)
    # Penalized group effects intentionally retain columns confounded with
    # the intercept; identification comes from their full-rank penalty.
    assert np.isfinite(X).all()


def test_stage_6_solution_and_penalized_normal_equations(smooth_pipeline):
    case = smooth_pipeline
    solution = case.model.gam_result_.fit_core_solution
    assert isinstance(solution, FitCoreSolution)
    fit = solution.fit_result
    assert fit.converged
    X = case.model.lpmatrix(case.data)
    assert fit.coef_full.shape == (X.shape[1],)
    assert np.isfinite(fit.coef_full).all() and np.isfinite(fit.cov_bayes).all()
    np.testing.assert_allclose(
        X @ fit.coef_full + case.data.off, fit.eta, atol=2e-9, rtol=2e-9
    )
    if case.basis in REGULAR_BASES:
        # Independent dense weighted penalized least squares, assembled from
        # the public design and each compiled block (not the fitting solver).
        penalty = np.zeros((X.shape[1], X.shape[1]))
        mapping = case.compiled.coef_reduced_to_full_idx
        for block in case.compiled.compiled_penalties:
            indices = mapping[block.coef_slice]
            penalty[np.ix_(indices, indices)] += (
                case.model.smoothing_params[block.smoothing_index] * block.matrix
            )
        w = case.data.w.to_numpy()
        normal = X.T @ (w[:, None] * X) + penalty
        rhs = X.T @ (w * (case.data.y - case.data.off))
        beta = np.linalg.solve(normal, rhs)
        np.testing.assert_allclose(X @ beta, X @ fit.coef_full, atol=2e-8, rtol=2e-8)
        covariance = np.linalg.inv(normal) * fit.scale
        np.testing.assert_allclose(
            X @ covariance @ X.T, X @ fit.cov_bayes @ X.T, atol=2e-8, rtol=2e-8
        )
    else:
        mask = fit.positive_coefficient_mask
        assert mask.any()
        assert (fit.coef_full[mask] > 0).all()


def test_stage_7_public_components_uncertainty_and_persistence(
    smooth_pipeline, tmp_path
):
    case = smooth_pipeline
    estimator = case.estimator
    X = case.model.lpmatrix(case.newdata)
    prediction = estimator.predict(case.newdata)
    np.testing.assert_allclose(
        prediction,
        X @ case.model.fit_result().coef_full + case.newdata.off,
        atol=2e-9,
        rtol=2e-9,
    )
    components = estimator.predict_components(case.newdata)
    components.validate_additive_reconstruction(rtol=1e-9, atol=2e-9)
    se = estimator.standard_errors(case.newdata)
    kernel = X @ case.model.vcov() @ X.T
    np.testing.assert_allclose(se**2, np.diag(kernel), atol=2e-9, rtol=2e-9)
    order = [6, 2, 4, 0, 5, 1, 3]
    np.testing.assert_allclose(
        estimator.predict(case.newdata.iloc[order]),
        prediction[order],
        atol=2e-9,
        rtol=2e-9,
    )
    path = estimator.save_model(tmp_path / (case.name + ".nampy"))
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(case.newdata), prediction)
    restored.predict_components(case.newdata).validate_additive_reconstruction()


@pytest.mark.parametrize(
    "code", REGULAR_BASES + MULTIVARIATE_ADAPTIVE_CASES + SHAPE_BASES
)
def test_pipeline_observable_invariants_match_upstream(code):
    from tests.parity._pipeline_invariants import assert_reference_invariants
    from tests.parity._registered_pipeline_reference import pipeline_reference
    from tests.parity._seven_stage_cases import smooth_inputs

    case = fit_smooth_case(code)
    frame, _, formula, options = smooth_inputs(code)
    engine = "scam" if code in SHAPE_BASES else "mgcv"
    if engine == "mgcv":
        # sc/scad without inequality requests are the corresponding underlying
        # bases; their unsupported scasm inequality requests are tested separately.
        formula = formula.replace("bs='sc'", "bs='bs'").replace("bs='scad'", "bs='ad'")
    else:
        from tests.scam.test_scam_fixed_smoothing_fit_parity import _UPSTREAM_BASIS_CODE

        formula = formula.replace(
            f"bs='{code}'", f"bs='{_UPSTREAM_BASIS_CODE.get(code, code)}'"
        )
    expected = pipeline_reference(
        code,
        case.data,
        case.newdata,
        [formula],
        "gaussian()",
        knots=options.get("knots"),
        engine=engine,
        sp=case.model.smoothing_params,
        start=None,
    )
    tolerance = 2e-5 if code == "cs" else 2e-7
    assert_reference_invariants(
        case.model, case.data, case.newdata, expected, tolerance=tolerance
    )


@pytest.mark.parametrize(
    "code", ["cs", "tp", "ts", "ds", "gp", "sos", "mrf", "ad__2d", "t2"]
)
def test_nonunique_basis_outputs_survive_training_row_permutation(code):
    from sklearn.base import clone

    from tests.parity._pipeline_invariants import predictor_evaluation

    case = fit_smooth_case(code)
    permuted = case.data.sample(frac=1, random_state=917)
    other = (
        clone(case.estimator).fit(permuted, sample_weight=permuted.w.to_numpy()).gam_
    )
    tolerance = 2e-5 if code == "cs" else 2e-7
    for name in ("aic", "bic", "loglik"):
        np.testing.assert_allclose(
            getattr(other, name)(),
            getattr(case.model, name)(),
            atol=tolerance,
            rtol=tolerance,
            err_msg=name,
        )
    np.testing.assert_allclose(
        other.fit_result().edf_total,
        case.model.fit_result().edf_total,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        other.predict(case.newdata),
        case.model.predict(case.newdata),
        atol=tolerance,
        rtol=tolerance,
    )
    left, right = (
        predictor_evaluation(case.model, case.newdata),
        predictor_evaluation(other, case.newdata),
    )
    np.testing.assert_allclose(
        left @ case.model.vcov() @ left.T,
        right @ other.vcov() @ right.T,
        atol=tolerance,
        rtol=tolerance,
    )


def test_reference_captures_and_frozen_inputs_match_provenance():
    import json

    from tests.parity.generate_registered_pipeline_manifest import DIRECTORY, manifest

    assert manifest() == json.loads((DIRECTORY / "manifest.json").read_text())
    responses = json.loads((DIRECTORY / "smooth_responses.json").read_text())
    assert set(responses) == set(SHAPE_BASES)
    assert all(
        len(values) == 96 and np.isfinite(values).all() for values in responses.values()
    )
