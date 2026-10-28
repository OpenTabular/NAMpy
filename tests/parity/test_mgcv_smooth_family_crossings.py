"""Non-Gaussian seven-stage backfill for every regular mgcv smooth.

These are fixed-smoothing conditional comparisons. They complement the
Gaussian registry matrix and the independently tested selection routes.
"""

import itertools
import pickle

import numpy as np
import pytest
from scipy.special import expit

from nampy import GAMClassifier, GAMRegressor
from tests.parity import test_registered_smooth_seven_stage_pipeline as stages
from tests.parity._pipeline_invariants import assert_reference_invariants
from tests.parity._registered_pipeline_reference import pipeline_reference
from tests.parity._seven_stage_cases import REGULAR_BASES, PipelineCase, smooth_inputs

pytestmark = [pytest.mark.surface_regression]
FAMILIES = ("binomial", "poisson", "gamma")
CROSSINGS = tuple(itertools.product(REGULAR_BASES, FAMILIES))
STAGE_CHECKS = (
    stages.test_stage_1_formula_spec_and_offset,
    stages.test_stage_2_runtime_basis_and_penalty_ownership,
    stages.test_stage_3_constructed_term_retains_penalty_and_prediction,
    stages.test_stage_4_compiler_blocks_and_smoothing_indices,
    stages.test_stage_5_identifiability_and_newdata_pairing,
)


def fit_crossing_case(code, family):
    frame, newdata, formula, options = smooth_inputs(code)
    rng = np.random.default_rng(281028)
    eta = 0.15 + 0.35 * np.sin(3 * frame.x) - 0.2 * frame.z + frame.off
    if family == "binomial":
        frame["y"] = rng.binomial(1, expit(eta))
        cls, specification, source = GAMClassifier, family, "binomial()"
    elif family == "poisson":
        frame["y"] = rng.poisson(np.exp(eta))
        cls, specification, source = GAMRegressor, family, "poisson()"
    else:
        frame["y"] = rng.gamma(5, np.exp(eta) / 5)
        cls, specification, source = (
            GAMRegressor,
            {"name": "gamma", "link": "log"},
            "Gamma(link='log')",
        )
    estimator = cls(
        family=specification,
        formula=formula,
        smoothing_params=0.4,
        smoothing_method="fixed",
        optimize_smoothing=False,
        max_irls_iter=300,
        irls_tol=1e-10,
        **options,
    ).fit(frame, frame.y.to_numpy(), sample_weight=frame.w.to_numpy())
    case = PipelineCase(code, frame, newdata, estimator, (formula,))
    source_formula = formula.replace("bs='sc'", "bs='bs'").replace(
        "bs='scad'", "bs='ad'"
    )
    return case, source_formula, source, options.get("knots")


@pytest.fixture(scope="module", params=CROSSINGS, ids=lambda p: "-".join(p))
def crossing(request):
    code, family = request.param
    case, source_formula, source, knots = fit_crossing_case(code, family)
    expected = pipeline_reference(
        code + "-" + family,
        case.data,
        case.newdata,
        [source_formula],
        source,
        knots=knots,
        sp=case.model.smoothing_params,
        namespace="mgcv_smooth_family_crossings",
    )
    return case, family, expected


@pytest.mark.parametrize("stage", range(1, 6), ids=lambda i: f"stage-{i}")
def test_stage_1_through_5_non_gaussian_ownership(crossing, stage):
    case, _, _ = crossing
    STAGE_CHECKS[stage - 1](case)


def test_stage_6_non_gaussian_observable_invariants(crossing):
    case, _, expected = crossing
    assert case.model.gam_result_.fit_core_solution.fit_result.converged
    assert_reference_invariants(
        case.model,
        case.data,
        case.newdata,
        expected,
        tolerance=2e-5 if case.basis == "cs" else 2e-7,
    )


def test_stage_7_non_gaussian_public_outputs_and_persistence(crossing):
    case, family, _ = crossing
    case.estimator.predict_components(case.newdata).validate_additive_reconstruction()
    link, se = case.model.predict(case.newdata, type="link", return_se=True)
    response, response_se = case.model.predict(
        case.newdata, type="response", return_se=True
    )
    assert np.isfinite(response).all() and np.isfinite(response_se).all()
    assert (response_se >= 0).all()
    inverse_derivative = case.model.family.inverse_link_derivatives(link, 1)
    np.testing.assert_allclose(
        response_se, np.abs(inverse_derivative) * se, atol=2e-10, rtol=2e-10
    )
    if family == "binomial":
        assert ((response >= 0) & (response <= 1)).all()
        np.testing.assert_allclose(
            case.estimator.predict_proba(case.newdata)[:, 1], response
        )
    else:
        assert (response > 0).all()
        np.testing.assert_allclose(case.estimator.predict(case.newdata), response)
    order = [6, 2, 4, 0, 5, 1, 3]
    np.testing.assert_allclose(
        case.model.predict(case.newdata.iloc[order]), response[order]
    )
    restored = pickle.loads(pickle.dumps(case.estimator))
    np.testing.assert_array_equal(restored.gam_.predict(case.newdata), response)
    np.testing.assert_array_equal(
        restored.gam_.predict(case.newdata, type="link", return_se=True)[1], se
    )


def test_complete_non_gaussian_crossings_have_reference_provenance():
    import json

    from tests.parity.generate_registered_pipeline_manifest import DIRECTORY, manifest

    actual = manifest("mgcv_smooth_family_crossings")
    stored = json.loads(
        (DIRECTORY.parent / "mgcv_smooth_family_crossings/manifest.json").read_text()
    )
    assert actual == stored
    assert {record["case"] for record in actual["captures"].values()} == {
        code + "-" + family for code, family in CROSSINGS
    }
