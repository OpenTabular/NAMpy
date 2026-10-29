"""BCCG analytic REML derivatives and automatic smoothing endpoints."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy import GAMLSS
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.model_state import _fit_result
from tests.extensions.test_bccg_fitting import _data


def _model(**kwargs):
    return GAMLSS(
        family="bccg_lss",
        formula={
            "mu": "y~s(x,bs='cr',k=5)",
            "sigma": "~s(x,bs='cr',k=4)",
            "nu": "~s(x,bs='cr',k=4)",
        },
        **kwargs,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_outer_reml_gradient_hessian_with_three_penalties_and_offsets(weighted):
    frame = _data(sigma=0.18, nu=-3.0)
    weights = np.resize([1.0, 0.0, 2.0, 0.5, 3.0], len(frame)) if weighted else None
    x = frame.x.to_numpy()
    offsets = [0.04 * np.sin(x), -0.03 * x, 0.08 * x]
    model = _model(optimize_smoothing=False, smoothing_params=[0.3, 0.5, 0.7], irls_tol=1e-10).fit(frame, sample_weight=weights, offset=np.column_stack(offsets))
    gam = model.gam_
    rho = np.log(gam.smoothing_params)

    def value(r):
        return np.array([criterion_value(gam, gam.y_, r, method="reml")])

    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    provenance = gam._general_family_outer_derivative_info
    assert provenance["gradient_source"] == provenance["hessian_source"] == "analytic"
    assert provenance["penalty_logdet_source"] == "analytic"
    assert "fallback_reason" not in provenance
    assert_allclose(
        gradient,
        approx_derivative(value, rho, abs_step=2e-4).ravel(),
        atol=2e-4,
        rtol=2e-4,
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda r: criterion_gradient(gam, gam.y_, r, method="reml"),
            rho,
            abs_step=2e-4,
        ),
        atol=2e-3,
        rtol=2e-3,
    )
    direction = np.array([1.0, -2.0, 1.0]) / np.sqrt(6)
    # Each value refits the coefficients. Unlike gradient differences, second
    # value differences amplify inner-solve error by 1 / step**2. Check two
    # larger steps to avoid cancellation and verify finite-difference stability.
    for step in (0.02, 0.04):
        second = (
            value(rho + step * direction)[0]
            - 2 * value(rho)[0]
            + value(rho - step * direction)[0]
        ) / step**2
        assert_allclose(direction @ hessian @ direction, second, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_automatic_reml_all_three_predictors_smoothed(optimizer):
    frame = _data()
    model = _model(smoothing_optimizer=optimizer).fit(frame)
    assert _fit_result(model.gam_).converged, _fit_result(model.gam_).failure_reason
    assert model.gam_._optim_method == "reml"
    assert np.all(np.isfinite(model.gam_.smoothing_params))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
    assert np.isfinite(model.score(frame, frame.y))
    assert model.standard_errors(frame).shape == (len(frame), 3)
    assert model.distribution_fit_status().eligible, model.distribution_fit_status()


@pytest.fixture(scope="module")
def independent_reml_profile():
    from functools import cache

    from scipy.optimize import minimize_scalar

    frame = _data()
    frame["y"] *= np.exp(0.5 * np.sin(np.pi * frame.x))
    formula = {"mu": "y~s(x,bs='cr',k=6)", "sigma": "~1", "nu": "~1"}

    @cache
    def profile(rho):
        fit = GAMLSS(
            family="bccg_lss",
            formula=formula,
            optimize_smoothing=False,
            smoothing_params=[np.exp(rho)],
            irls_tol=1e-9,
        ).fit(frame)
        score = criterion_value(fit.gam_, fit.gam_.y_, np.array([rho]), method="reml")
        return fit, score

    optimum = minimize_scalar(
        lambda r: profile(float(r))[1],
        bounds=(-5.0, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    assert optimum.success and -4.99 < optimum.x < 9.99
    assert profile(optimum.x - 0.1)[1] > optimum.fun
    assert profile(optimum.x + 0.1)[1] > optimum.fun
    return frame, formula, optimum, profile(float(optimum.x))[0]


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_automatic_endpoint_matches_independent_derivative_free_reml_profile(
    optimizer, independent_reml_profile, tmp_path
):
    frame, formula, optimum, reference = independent_reml_profile
    model = GAMLSS(
        family="bccg_lss",
        formula=formula,
        smoothing_optimizer=optimizer,
        sp_log_bounds=(-5.0, 10.0),
    ).fit(frame)
    assert _fit_result(model.gam_).converged, (
        _fit_result(model.gam_).failure_reason,
        np.log(model.gam_.smoothing_params),
        optimum.x,
    )
    rho = np.log(model.gam_.smoothing_params)
    value = criterion_value(model.gam_, model.gam_.y_, rho, method="reml")
    assert_allclose(value, optimum.fun, atol=2e-5, rtol=0)
    assert_allclose(rho, [optimum.x], atol=5e-3, rtol=0)
    assert_allclose(model.predict_parameters(frame), reference.predict_parameters(frame), atol=2e-4, rtol=0)
    assert (
        np.max(
            np.abs(criterion_gradient(model.gam_, model.gam_.y_, rho, method="reml"))
        )
        < 2e-3
    )
    model.predict_components(frame).validate_additive_reconstruction()
    assert model.distribution_fit_status().eligible
    restored = GAMLSS.load_model(model.save_model(tmp_path / "bccg_auto.pkl"))
    assert_allclose(
        restored.predict_distribution(frame).quantile([0.1, 0.5, 0.9]),
        model.predict_distribution(frame).quantile([0.1, 0.5, 0.9]),
    )
    assert _fit_result(model.gam_).cov_unconditional is not None
    assert np.isfinite(model.gam_.vcov(unconditional=True)).all()


def test_tight_inner_tolerance_keeps_nonconvergence_status_without_optimizer_rescue(
    independent_reml_profile,
):
    frame, formula, optimum, _ = independent_reml_profile
    model = GAMLSS(
        family="bccg_lss", formula=formula, sp_log_bounds=(-5.0, 10.0), irls_tol=1e-9
    ).fit(frame)
    status = model.distribution_fit_status()
    if not _fit_result(model.gam_).converged:
        assert status.state == "nonconverged" and not status.eligible
        assert _fit_result(model.gam_).failure_reason
    assert_allclose(np.log(model.gam_.smoothing_params), [optimum.x], atol=5e-3, rtol=0)


def test_runnable_reference_example_uses_eligible_automatic_reml():
    import runpy
    from pathlib import Path

    path = (
        Path(__file__).parents[2]
        / "docs"
        / "_examples"
        / "reference"
        / "statistical_reference"
        / "families"
        / "bccg_lss.py"
    )
    namespace = runpy.run_path(str(path))
    model = namespace["model"]
    assert model.optimize_smoothing
    assert model.gam_._optim_method == "reml"
    assert model.distribution_fit_status().eligible, model.distribution_fit_status()
