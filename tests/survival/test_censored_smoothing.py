"""Analytic censored REML derivatives and independent smoothing endpoints."""

from functools import cache

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar
from scipy.optimize._numdiff import approx_derivative

from nampy import SurvivalGAM
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.survival import SurvivalTarget
from tests.survival.test_censored_derivatives import reference_loglik


def independent_laplace_value(model, name, target, weights, offsets):
    """gam.fit5 Laplace value from independent likelihood curvature, gamma=1."""
    import torch

    core = model.gam_.gam_result_.fit_core_solution
    matrix = torch.tensor(core.fit_state.X)
    penalty = np.asarray(core.fit_state.penalty_matrix)
    beta = torch.tensor(core.fit_result.coef_full)
    jj = model.gam_.gam_result_.require_compiled_model().predictor_full_indices
    weight = (
        torch.ones(len(target), dtype=torch.float64)
        if weights is None
        else torch.tensor(weights)
    )

    def likelihood(b):
        eta = torch.stack([matrix[:, block] @ b[block] for block in jj], dim=1)
        return (
            reference_loglik(name, eta + torch.tensor(offsets), target) * weight
        ).sum()

    curvature = -torch.func.hessian(likelihood)(beta).numpy() + penalty
    sign, logdet = np.linalg.slogdet(curvature)
    assert sign > 0
    eigenvalues = np.linalg.eigvalsh(penalty)
    positive = eigenvalues > eigenvalues.max() * np.finfo(float).eps * 512
    nullity = len(beta) - positive.sum()
    return (
        -likelihood(beta).item()
        + 0.5 * beta.numpy() @ penalty @ beta.numpy()
        + 0.5 * logdet
        - 0.5 * np.log(eigenvalues[positive]).sum()
        - 0.5 * nullity * np.log(2 * np.pi)
    )


def smooth_sample(name):
    rng = np.random.default_rng(286)
    x = rng.uniform(-1.0, 1.0, 180)
    location = 0.3 + 0.65 * np.sin(np.pi * x)
    second = 0.12 * np.cos(np.pi * x)
    latent = (
        np.exp(location) * rng.weibull(2.1 * np.exp(second))
        if name == "weibull"
        else np.exp(location + 0.5 * np.exp(second) * rng.normal(size=len(x)))
    )
    entry = rng.uniform(0.03, 0.25, len(x))
    entry[::5] = 0.0
    selected = latent > entry
    x, latent, entry = x[selected], latent[selected], entry[selected]
    censor = rng.uniform(1.2, 2.8, len(x))
    return pd.DataFrame({"x": x}), SurvivalTarget(
        np.minimum(latent, censor), latent <= censor, entry
    )


def formulas(name, *, both=True):
    names = ("scale", "shape") if name == "weibull" else ("loc", "scale")
    return {
        names[0]: "~s(x,bs='cr',k=6)",
        names[1]: "~s(x,bs='cr',k=4)" if both else "~1",
    }


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
@pytest.mark.parametrize("weighted", [False, True])
def test_outer_reml_gradient_hessian_with_censoring_entry_and_offsets(name, weighted):
    X, target = smooth_sample(name)
    weights = np.resize([1.0, 0.0, 2.0, 0.5, 3.0], len(X)) if weighted else None
    offsets = np.column_stack([0.04 * np.sin(X.x), -0.03 * X.x])
    model = SurvivalGAM(
        likelihood=name,
        formula=formulas(name),
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.7],
        irls_tol=1e-9,
    ).fit(X, target, sample_weight=weights, offset=offsets)
    gam = model.gam_
    assert_allclose(
        criterion_value(gam, gam.y_, np.log(gam.smoothing_params), method="reml"),
        independent_laplace_value(model, name, target, weights, offsets),
        atol=2e-7,
        rtol=0,
    )
    for rho in (np.log([0.3, 0.7]), np.log([0.8, 1.2])):

        def value(r):
            return np.array([criterion_value(gam, gam.y_, r, method="reml")])

        gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
        hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
        provenance = gam._general_family_outer_derivative_info
        assert (
            provenance["gradient_source"] == provenance["hessian_source"] == "analytic"
        )
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
        direction = np.array([1.0, -2.0]) / np.sqrt(5)
        step = 0.005
        second = (
            value(rho + step * direction)[0]
            - 2 * value(rho)[0]
            + value(rho - step * direction)[0]
        ) / step**2
        assert_allclose(direction @ hessian @ direction, second, atol=3e-3, rtol=3e-3)


@pytest.fixture(scope="module", params=["weibull", "lognormal"])
def independent_censored_profile(request):
    name = request.param
    X, target = smooth_sample(name)
    formula = formulas(name, both=False)

    @cache
    def profile(rho):
        model = SurvivalGAM(
            likelihood=name,
            formula=formula,
            optimize_smoothing=False,
            smoothing_params=[np.exp(rho)],
        ).fit(X, target)
        value = criterion_value(
            model.gam_, model.gam_.y_, np.array([rho]), method="reml"
        )
        return model, value

    optimum = minimize_scalar(
        lambda r: profile(float(r))[1],
        bounds=(-5.0, 8.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    assert optimum.success and -4.99 < optimum.x < 7.99
    assert profile(optimum.x - 0.1)[1] > optimum.fun
    assert profile(optimum.x + 0.1)[1] > optimum.fun
    return name, X, target, formula, optimum, profile(float(optimum.x))[0]


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs"])
def test_automatic_reml_two_smooth_parameters(name, optimizer):
    from nampy.gam.diagnostics.fit_status import distribution_fit_status

    X, target = smooth_sample(name)
    model = SurvivalGAM(
        likelihood=name, formula=formulas(name), smoothing_optimizer=optimizer
    ).fit(X, target)
    assert model.gam_._optim_method == "reml"
    assert distribution_fit_status(model.gam_).eligible, distribution_fit_status(
        model.gam_
    )
    assert np.isfinite(model.gam_.smoothing_params).all()
    assert (np.asarray(model.gam_.smoothing_params) > 0).all()
    assert np.isfinite(model.score(X, target))
    assert np.isfinite(model.gam_.vcov()).all()


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs"])
def test_automatic_endpoint_matches_independent_reml_profile(
    optimizer, independent_censored_profile, tmp_path
):
    from nampy.gam.diagnostics.fit_status import distribution_fit_status

    name, X, target, formula, optimum, reference = independent_censored_profile
    model = SurvivalGAM(
        likelihood=name, formula=formula, smoothing_optimizer=optimizer
    ).fit(X, target)
    assert distribution_fit_status(model.gam_).eligible, distribution_fit_status(
        model.gam_
    )
    rho = np.log(model.gam_.smoothing_params)
    score = criterion_value(model.gam_, model.gam_.y_, rho, method="reml")
    assert_allclose(score, optimum.fun, atol=2e-5, rtol=0)
    assert_allclose(rho, [optimum.x], atol=5e-3, rtol=0)
    assert_allclose(model.predict_link(X), reference.predict_link(X), atol=2e-4, rtol=0)
    assert (
        np.max(
            np.abs(criterion_gradient(model.gam_, model.gam_.y_, rho, method="reml"))
        )
        < 2e-3
    )
    model.predict_components(X).validate_additive_reconstruction()
    assert model.gam_.fit_result().core.cov_unconditional is not None
    assert np.isfinite(model.gam_.vcov(unconditional=True)).all()
    restored = SurvivalGAM.load_model(
        model.save_model(tmp_path / (name + optimizer + ".nampy"))
    )
    assert_allclose(
        restored.predict(X, times=[0, 0.5, 1, 2, 4]),
        model.predict(X, times=[0, 0.5, 1, 2, 4]),
        atol=0,
        rtol=0,
    )
    assert_allclose(
        restored.gam_.vcov(unconditional=True),
        model.gam_.vcov(unconditional=True),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_unsupported_criteria_and_optimizers_stay_explicit(name):
    X, target = smooth_sample(name)
    for method in ("ml", "laml", "gcv", "ubre", "ncv"):
        with pytest.raises(NotImplementedError, match="Separate censored"):
            SurvivalGAM(likelihood=name, smoothing_method=method).fit(X, target)
    for optimizer in ("efs", "optim", "magic", "nlm.fd"):
        with pytest.raises(NotImplementedError, match="no validated smoothing route"):
            SurvivalGAM(likelihood=name, smoothing_optimizer=optimizer).fit(X, target)

    from nampy import GAMLSS
    from nampy.gam.families.gamlss.censored import CensoredLogNormal, CensoredWeibull

    family = CensoredWeibull() if name == "weibull" else CensoredLogNormal()
    formula = dict(formulas(name, both=False))
    first = family.parameter_names[0]
    formula[first] = "y " + formula[first]
    with pytest.raises(NotImplementedError, match="no validated smoothing route"):
        GAMLSS(family=family, formula=formula, smoothing_optimizer="efs").fit(
            X.assign(y=target.time),
            context={"event": target.event, "entry": target.entry},
        )
