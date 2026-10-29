"""Censored endpoints, delayed entry, fixed smoothing and explicit boundaries."""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.survival import SurvivalTarget
from tests.survival.test_censored_derivatives import reference_loglik


def sample(name, n=110):
    rng = np.random.default_rng(81)
    x = rng.uniform(-1.0, 1.0, n)
    latent = (
        np.exp(0.2 + 0.35 * x) * rng.weibull(np.exp(0.3 - 0.15 * x))
        if name == "weibull"
        else np.exp(0.2 + 0.35 * x + np.exp(-0.4 + 0.1 * x) * rng.normal(size=n))
    )
    entry = rng.uniform(0.0, 0.35, n)
    active = latent > entry
    x, latent, entry = x[active], latent[active], entry[active]
    censor = rng.uniform(0.7, 2.7, len(x))
    target = SurvivalTarget(np.minimum(latent, censor), latent <= censor, entry)
    X = pd.DataFrame({"x": x})
    weights = np.resize(np.array([1.0, 0.0, 2.0, 1.5]), len(x))
    return X, target, weights


def independent_endpoint(name, X, target, weights, offsets=None):
    design = torch.tensor(np.column_stack([np.ones(len(X)), X.x]), dtype=torch.float64)
    offsets = (
        torch.zeros((len(X), 2), dtype=torch.float64)
        if offsets is None
        else torch.tensor(offsets)
    )

    def objective(beta):
        coefficients = torch.tensor(beta, requires_grad=True)
        eta = design @ coefficients.reshape(2, 2).T + offsets
        loss = -(reference_loglik(name, eta, target) * torch.tensor(weights)).sum()
        (gradient,) = torch.autograd.grad(loss, coefficients)
        return loss.item(), gradient.detach().numpy()

    result = minimize(
        objective,
        np.array([0.2, 0.2, -0.2 if name == "lognormal" else 0.2, 0.0]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 500},
    )
    assert np.max(np.abs(result.jac)) < 2e-5
    return (design @ torch.tensor(result.x).reshape(2, 2).T + offsets).numpy()


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_censored_statistical_endpoint_matches_independent_weighted_likelihood(
    name, tmp_path
):
    X, y, weights = sample(name)
    names = ("scale", "shape") if name == "weibull" else ("loc", "scale")
    offsets = np.column_stack([0.05 * np.sin(X.x), 0.03 * np.cos(X.x)])
    model = SurvivalGAM(
        likelihood=name, formula=dict.fromkeys(names, "~x"), optimize_smoothing=False
    ).fit(X, y, sample_weight=weights, offset=offsets)
    expected = independent_endpoint(name, X, y, weights, offsets)
    assert_allclose(model.predict_link(X, offset=offsets), expected, atol=8e-6, rtol=0)
    assert_allclose(
        model.score(X, y, weights, offset=offsets) * weights.sum(),
        model.survival_log_likelihood_,
        atol=2e-10,
    )
    components = model.predict_components(X, offset=offsets)
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(y), 2)
    assert_allclose(components.link, model.predict_link(X, offset=offsets))
    restored = SurvivalGAM.load_model(model.save_model(tmp_path / (name + ".nampy")))
    assert_allclose(
        restored.predict(X, times=[0, 0.5, 1, 3], offset=offsets),
        model.predict(X, times=[0, 0.5, 1, 3], offset=offsets),
        rtol=0,
        atol=0,
    )
    before = model.predict(X, times=[1], offset=offsets)
    with pytest.raises(ValueError, match="positive-weight event"):
        model.fit(X, SurvivalTarget(y.time, np.zeros(len(y))), offset=offsets)
    assert_allclose(model.predict(X, times=[1], offset=offsets), before)
    with pytest.raises(ValueError, match="offset"):
        model.predict(X, times=[1])


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_fixed_smooth_two_parameters_frequency_weights_and_covariance(name):
    X, y, _ = sample(name)
    weights = np.resize([1, 0, 2], len(y))
    names = ("scale", "shape") if name == "weibull" else ("loc", "scale")
    knots = {"x": np.linspace(-1, 1, 4)}
    model = SurvivalGAM(
        likelihood=name,
        formula=dict.fromkeys(names, "~s(x,bs='cr',k=4)"),
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.6],
        knots=knots,
        irls_tol=1e-9,
    ).fit(X, y, sample_weight=weights)
    rows = np.repeat(np.arange(len(y)), weights)
    # Hold the compiled basis/penalty fixed. Recompilation on duplicated
    # subjects can change mgcv centering and penalty scaling even at fixed sp.
    core = model.gam_.gam_result_.fit_core_solution
    matrix = core.fit_state.X
    penalty = core.fit_state.penalty_matrix
    beta = core.fit_result.coef_full
    jj = model.gam_.gam_result_.require_compiled_model().predictor_full_indices
    family = model.gam_.family
    repeated = type(family)()
    repeated.bind_fit_context(
        {"event": y.event[rows], "entry": y.entry[rows]}, len(rows)
    )
    weighted = family.ll(y.time, matrix, jj, beta, weights, deriv=1)
    duplicated = repeated.ll(
        y.time[rows], matrix[rows], jj, beta, np.ones(len(rows)), deriv=1
    )
    for key in ("l", "lb", "lbb"):
        assert_allclose(weighted[key], duplicated[key], atol=2e-11)

    def objective(b):
        tensor = torch.tensor(b, requires_grad=True)
        mat = torch.tensor(matrix)
        eta = torch.stack([mat[:, block] @ tensor[block] for block in jj], dim=1)
        loss = (
            -(reference_loglik(name, eta, y) * torch.tensor(weights)).sum()
            + 0.5 * tensor @ torch.tensor(penalty) @ tensor
        )
        (gradient,) = torch.autograd.grad(loss, tensor)
        return loss.item(), gradient.detach().numpy()

    reference = minimize(
        objective, beta + 0.03, method="BFGS", jac=True, options={"gtol": 1e-8}
    )
    assert np.max(np.abs(reference.jac)) < 2e-5
    for block in jj:
        assert_allclose(
            matrix[:, block] @ beta[block],
            matrix[:, block] @ reference.x[block],
            atol=8e-6,
            rtol=0,
        )
    assert np.isfinite(model.gam_.vcov()).all()
    assert np.linalg.eigvalsh(model.gam_.vcov()).min() > 0
    assert np.isfinite(model.parametric_.standard_errors(X)).all()
    with pytest.raises(NotImplementedError, match="Separate censored"):
        SurvivalGAM(likelihood=name, smoothing_method="gcv").fit(X, y)


@pytest.mark.parametrize("name", ["weibull", "lognormal"])
def test_neural_censored_endpoint_and_shared_curves(name, tmp_path):
    X, y, weights = sample(name)
    training = GradientTrainingConfig(
        max_epochs=1500,
        lr=0.025,
        batch_size=len(y),
        weight_decay=0,
        average_checkpoints=True,
        lr_schedule="none",
        patience=1500,
        checkpoint_path=str(tmp_path),
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    model = LinRegSurvival(
        objective=SurvivalObjectiveConfig(likelihood=name), training=training
    )
    model.fit(
        X, y, X_val=X.copy(), y_val=y, sample_weight=weights, sample_weight_val=weights
    )
    expected = independent_endpoint(name, X, y, weights)
    assert_allclose(model.predict_link(X), expected, atol=2e-3, rtol=0)
    components = model.predict_components(X, center=True)
    # The record constructor checks reconstruction at the source precision.
    assert components.intercept.dtype == np.float32
    assert_allclose(components.link, model.predict_link(X), atol=0, rtol=0)
    curves = model.predict(X, times=[0, 0.5, 1.0, 3.0])
    assert_allclose(curves[:, 0], 1.0)
    assert (np.diff(curves, axis=1) <= 0).all()
    assert np.isfinite(model.score(X, y, weights))
    restored = LinRegSurvival.load_model(model.save_model(tmp_path / (name + ".nampy")))
    assert_allclose(
        restored.predict(X, times=[0, 0.5, 1.0, 3.0]), curves, rtol=0, atol=0
    )
