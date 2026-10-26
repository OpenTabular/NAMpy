"""SCAT outer control and postfit availability contracts from mgcv."""

from dataclasses import replace

import numpy as np
import pytest

from nampy.gam.fit.postprocess.unconditional_covariance import (
    apply_unconditional_postfit,
)
from nampy.gam.model_state import _fit_result, _fit_workspace
from tests.families.test_mgcv_scat_workflows import _case, _fit


def test_scat_vector_newton_forwards_all_controls(monkeypatch):
    from nampy.gam.fit.selection.optimize import driver

    original = driver.optimize_outer_newton_indefinite_hessian
    seen = []

    def capture(**kwargs):
        seen.append(kwargs.copy())
        return original(**kwargs)

    monkeypatch.setattr(driver, "optimize_outer_newton_indefinite_hessian", capture)
    case = _case(estimated=True)
    case["control"] = {
        "newton": {
            "conv_tol": 2e-7,
            "max_n_step": 2.5,
            "max_s_step": 0.75,
            "max_half": 16,
        }
    }
    model = _fit(case)
    assert len(seen) == 1
    assert seen[0]["conv_tol"] == 2e-7
    assert seen[0]["max_nstep"] == 2.5
    assert seen[0]["max_sstep"] == 0.75
    assert seen[0]["max_half"] == 16
    assert np.isfinite(model.predict()).all()


def test_scat_explicit_scale_reaches_inner_and_outer_stopping(monkeypatch):
    from nampy.gam.fit.selection.criteria.pirls import family_vector_theta

    original = family_vector_theta.solve_pirls_given_smoothing
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["scale_reference"])
        return original(*args, **kwargs)

    monkeypatch.setattr(family_vector_theta, "solve_pirls_given_smoothing", capture)
    model = _fit(_case(estimated=False, method="REML", scale=2))
    assert seen and all(scale == 2 for scale in seen)
    assert _fit_workspace(model).pirls_outer_scale_est == 2


def test_scat_nlm_has_no_postfit_smoothing_correction():
    model = _fit(_case(estimated=False))
    result = _fit_result(model)
    seeded = replace(
        result,
        cov_unconditional=np.asarray(result.cov_bayes).copy(),
        edf2=np.ones(result.coef_full.size),
    )
    model.smoothing_optimizer = "nlm"
    actual = apply_unconditional_postfit(model, None, seeded, None)
    assert actual.cov_unconditional is None
    assert actual.edf2 is None


def test_scat_initial_spg_uses_observed_weights_and_global_fisher_fallback(monkeypatch):
    from types import SimpleNamespace

    from nampy.gam.families.scat import ScaledTFamily
    from nampy.gam.fit.selection import reparam
    from nampy.gam.fit.selection.optimize import basics

    y = np.array([0.0, 0.5, 1.0])
    prior = np.array([1.0, 0.0, 2.0])
    X = np.column_stack([np.ones(3), np.arange(3)])
    setup = SimpleNamespace(X=X, S=[np.eye(2)], off=[0], L=None, lsp0=[0])
    monkeypatch.setattr(basics, "_penalty_blocks_seq", lambda model: [object()])
    monkeypatch.setattr(basics, "_n_smoothing_params", lambda model: 1)
    monkeypatch.setattr(basics, "is_transformed_coefficient_model", lambda model: False)
    monkeypatch.setattr(reparam, "build_estimate_gam_setup_state", lambda model: setup)
    weighted_designs = []

    def capture(design, *args, **kwargs):
        weighted_designs.append(design)
        return np.ones(1)

    monkeypatch.setattr(
        basics, "_initial_smoothing_params_from_packed_penalties", capture
    )
    nu = 5.0
    for sigma in [1.0, 0.01]:
        family = ScaledTFamily(theta=[nu, sigma])
        model = SimpleNamespace(family=family, prior_weights_=prior)
        basics._initial_smoothing_params_from_design(model, y)
        if sigma == 1.0:
            residual = np.array([-0.1, 0, 0])
            denominator = nu * sigma**2 + residual**2
            expected = prior * (nu + 1) * (nu * sigma**2 - residual**2) / denominator**2
            assert expected[1] == 0  # no artificial positive-weight floor
        else:
            # A single negative observed weight switches every row, including
            # the zero-prior row: upstream SCAT EDmu2 is intentionally unweighted.
            expected = np.full(3, (nu + 1) / ((nu + 3) * sigma**2))
        np.testing.assert_allclose(weighted_designs[-1], np.sqrt(expected)[:, None] * X)


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_scat_positive_min_sp_is_explicitly_guarded_for_all_fit_routes(estimated, optimize):
    import pandas as pd

    from nampy.gam import GAM
    from nampy.gam.families.scat import ScaledTFamily

    case = _case(estimated=estimated)
    model = GAM(
        formula=case["formula"],
        family=ScaledTFamily(theta=case["theta"]),
        min_sp=[0.1],
        smoothing_method="REML" if optimize else "fixed",
        optimize_smoothing=optimize,
        smoothing_params=case["sp"],
    )
    with pytest.raises(NotImplementedError, match="scat with positive min_sp"):
        model.fit(data=pd.DataFrame(case["data"]))
