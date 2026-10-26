"""Separate the analytic postfit contract from outer stopping tolerances."""

import numpy as np
import pytest

from nampy.gam.fit.backends import solve_pirls_given_smoothing
from nampy.gam.fit.postprocess.extended_uncertainty import (
    extended_unconditional_postfit,
)
from nampy.gam.fit.solvers.stacked_qr import _signed_weight_rank_correction
from nampy.gam.fit.state import FitCoreSolution
from tests.families.test_mgcv_scat_workflows import (
    UNCERTAINTY_CASES,
    _case,
    _fit,
    _uncertainty_reference,
)


@pytest.mark.parametrize("index", [6, 7])
def test_scat_postfit_at_upstream_ancillary_and_smoothing_parameters(index):
    case = _case(**UNCERTAINTY_CASES[index])
    reference = _uncertainty_reference(case)
    model = _fit(case)
    model.family.putTheta(reference["theta"])
    model.smoothing_params = np.asarray(reference["sp"], dtype=float)
    # mgcv::sp.vcov(reg=1e-3), without edge correction, adds the scalar
    # regularizer to every Hessian entry, not just its diagonal (mgcv.r).
    model._optim_result.outer_info["hess"] = np.linalg.inv(reference["sp_vcov"]) - 0.001
    solution = solve_pirls_given_smoothing(model, model.y_, model.smoothing_params)
    core = (
        solution
        if isinstance(solution, FitCoreSolution)
        else FitCoreSolution.from_dict(solution)
    )
    covariance, edf2, _ = extended_unconditional_postfit(
        model, solution, core.fit_result, core.fit_state
    )
    for actual, key in [
        (core.fit_result.coef_full, "coef"),
        (core.fit_result.cov_bayes, "Vp"),
        (covariance, "Vc"),
        (edf2, "edf2"),
    ]:
        np.testing.assert_allclose(actual, reference[key], atol=2e-7, rtol=2e-7)


def test_gdipk_signed_pseudoinverse_retains_only_positive_curvature():
    # gdi.c::gdiPK sets 1-2*d^2 <= 0 to zero; pls_fit1 rejects the same
    # system for its positive-weight retry. The two contexts must differ.
    negative_rows = np.diag([0.9, 0.2])
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        _signed_weight_rank_correction(negative_rows, rank_tol=1e-12)
    correction, root, left, logdet, _ = _signed_weight_rank_correction(
        negative_rows, rank_tol=1e-12, signed_pseudoinverse=True
    )
    np.testing.assert_allclose(correction, np.diag([0, 1 / 0.92]), atol=1e-15)
    np.testing.assert_allclose(root.T @ root, correction, atol=1e-15)
    np.testing.assert_allclose(left.T @ left, np.diag([0, 0.92]), atol=1e-15)
    assert logdet == pytest.approx(np.log(0.92), abs=1e-15)


def test_gdipk_and_pls_fit1_agree_for_positive_signed_system():
    negative_rows = np.diag([0.2, 0.3])
    expected = np.diag(1 / (1 - 2 * np.array([0.2, 0.3]) ** 2))
    for pseudoinverse in [False, True]:
        correction, root, left, logdet, _ = _signed_weight_rank_correction(
            negative_rows, rank_tol=1e-12, signed_pseudoinverse=pseudoinverse
        )
        np.testing.assert_allclose(correction, expected, atol=1e-15)
        np.testing.assert_allclose(root.T @ root, expected, atol=1e-15)
        np.testing.assert_allclose(left.T @ left, np.linalg.inv(expected), atol=1e-15)
        assert logdet == pytest.approx(-np.linalg.slogdet(expected)[1], abs=1e-15)


@pytest.mark.parametrize("reject_start", [False, True])
def test_extended_pirls_warmstart_matches_gam_fit4_precedence(reject_start):
    from nampy.gam.families.scat import ScaledTFamily
    from nampy.gam.fit.solvers.irls_core import irls_core

    x = np.linspace(-1, 1, 20)
    X = np.column_stack([np.ones(x.size), x])
    y = 1 + 0.3 * x + 0.05 * np.cos(3 * x)
    null_coef = np.array([1.0, 0.0])
    good_start = np.array([1.0, 0.2])
    kwargs = {
        "X": X,
        "y": y,
        "family": ScaledTFamily(theta=[5, 1]),
        "S": np.diag([0.0, 0.1]),
        "null_coef": null_coef,
        "max_iter": 1,
    }
    if reject_start:
        # gam.fit4 discards coefficient, eta and mu starts together when
        # the supplied coefficients have larger penalized deviance than null.
        actual = irls_core(
            **kwargs,
            coef_start=np.array([100.0, 20.0]),
            etastart=np.full(x.size, 3.0),
            mustart=np.full(x.size, 2.0),
        )
        expected = irls_core(**kwargs)
    else:
        # Supplying both starts must have the same effect as coefficients
        # alone, even if mustart disagrees substantially.
        actual = irls_core(
            **kwargs, coef_start=good_start, mustart=np.full(x.size, 2.0)
        )
        expected = irls_core(**kwargs, coef_start=good_start)
    np.testing.assert_array_equal(actual["coef_full"], expected["coef_full"])
    assert actual["inner_trace"] == expected["inner_trace"]


def test_scat_positive_weight_retry_retains_supplied_weighted_response():
    import json
    from pathlib import Path

    from nampy.gam.fit.selection.criteria.pirls.family_vector_theta import (
        criterion_ml_reml_pirls_vector_theta,
    )

    root = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_scat_solver"
    reference = json.loads((root / "positive_weight_retry.json").read_text())
    theta = np.array([-0.44054612, -2.60251993])
    case = json.loads((root / "input.json").read_text())
    case["theta"] = [3 + np.exp(theta[0]), np.exp(theta[1])]
    model = _fit(case)
    from nampy.gam.model_state import _fit_result

    result = _fit_result(model)
    for expected in reference["result"]:  # upstream deriv=0 and deriv=2 agree
        np.testing.assert_allclose(
            result.coef_full, expected["coef"], atol=2e-7, rtol=2e-7
        )
        assert result.deviance == pytest.approx(expected["dev"], abs=2e-7)
        score = criterion_ml_reml_pirls_vector_theta(
            model, model.y_, np.log(model.smoothing_params), "REML"
        )
        assert score == pytest.approx(expected["score"], abs=2e-7)
