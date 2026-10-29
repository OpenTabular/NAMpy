"""Joint normal observation fitting, analytic derivatives and shared predictions."""

import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import truncnorm

from nampy import (
    GAMLSS,
    ObjectiveConfig,
    SemiStructuredLSS,
    SemiStructuredTrainingConfig,
)
from nampy.distributions import ObservationTarget, observation_log_prob
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.observations import NormalObservations
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.neural.configs import DefaultSemiStructuredConfig
from nampy.neural.families.observations import NormalObservationsDistribution


def data(n=160):
    rng = np.random.default_rng(271)
    X = pd.DataFrame({"x": np.linspace(-1, 1, n)})
    mu = 0.2 + 0.5 * np.sin(2 * X.x.to_numpy())
    y = truncnorm.rvs(
        (-0.5 - mu) / 0.7, (1.7 - mu) / 0.7, loc=mu, scale=0.7, random_state=rng
    )
    lower = y.copy()
    upper = y.copy()
    upper[y > 1.2] = np.inf
    lower[y > 1.2] = 1.2
    lower[y < -0.1] = -np.inf
    upper[y < -0.1] = -0.1
    interval = (y >= 0.3) & (y < 0.5)
    lower[interval] = 0.3
    upper[interval] = 0.5
    return X, ObservationTarget(lower, upper)


def test_analytic_coefficient_derivatives_and_neural_loss_agree():
    X, target = data(40)
    encoded = target.encoded
    family = NormalObservations(lower=-0.5, upper=1.7)
    family.bind_fit_context(
        {"observation_upper": encoded[:, 1], "observation_code": encoded[:, 2]},
        len(target),
    )
    predictors = np.tile([0.25, np.log(0.7)], (len(target), 1))
    variables = [Taylor(predictors[:, i], 2, 4, i) for i in range(2)]
    jet = family._log_density(encoded[:, 0], *variables)
    neural = NormalObservationsDistribution(lower=-0.5, upper=1.7)
    raw = torch.tensor(predictors, dtype=torch.float64, requires_grad=True)
    assert_allclose(
        jet.value,
        -neural.compute_loss(raw, torch.tensor(encoded), reduction="none").detach(),
        atol=1e-13,
    )
    assert torch.autograd.gradcheck(
        lambda value: neural.compute_loss(value, torch.tensor(encoded)),
        (raw,),
        atol=2e-5,
    )

    def ll(b):
        return -neural.compute_loss(
            b.expand(len(target), 2), torch.tensor(encoded), reduction="sum"
        )

    point = torch.tensor(predictors[0])
    for order, derivative in [
        (1, torch.func.grad(ll)),
        (2, torch.func.hessian(ll)),
        (3, torch.func.jacrev(torch.func.hessian(ll))),
        (4, torch.func.jacrev(torch.func.jacrev(torch.func.hessian(ll)))),
    ]:
        expected = derivative(point).numpy()
        for indices in np.ndindex(*(2,) * order):
            multi = tuple(indices.count(i) for i in range(2))
            assert_allclose(
                (
                    jet.coefficients[multi]
                    * __import__("math").factorial(multi[0])
                    * __import__("math").factorial(multi[1])
                ).sum(),
                expected[indices],
                rtol=3e-8,
                atol=2e-8,
            )


def test_fitted_statistical_endpoint_matches_independent_scipy_and_persists(tmp_path):
    X, target = data()
    family = NormalObservations(lower=-0.5, upper=1.7)
    model = GAMLSS(
        formula={"mean": "~1", "scale": "~1"},
        family=family,
        optimize_smoothing=False,
        irls_tol=1e-10,
    ).fit(X, target)

    # Independent SciPy response law; numerical derivatives occur only in this
    # reference optimizer, never in the production REML derivative path.
    def objective(b):
        source = truncnorm(
            (-0.5 - b[0]) / np.exp(b[1]),
            (1.7 - b[0]) / np.exp(b[1]),
            loc=b[0],
            scale=np.exp(b[1]),
        )
        lo, hi = target.lower, target.upper
        exact = lo == hi
        terms = np.empty(len(target))
        terms[exact] = source.logpdf(lo[exact])
        terms[~exact] = np.log(source.cdf(hi[~exact]) - source.cdf(lo[~exact]))
        return -terms.sum()

    expected = minimize(objective, [0.2, np.log(0.7)], method="BFGS", tol=1e-8)
    assert_allclose(model.predict_link(X)[0], expected.x, rtol=2e-6, atol=2e-6)
    prediction = model.predict_distribution(X)
    assert_allclose(
        -observation_log_prob(prediction, target).sum(), expected.fun, atol=1e-8
    )
    restored = model.load_model(model.save_model(tmp_path / "normal.nampy"))
    assert_allclose(
        restored.predict_distribution(X).mean(), prediction.mean(), rtol=0, atol=0
    )


def test_normal_observation_outer_derivatives_and_neural_fit(tmp_path):
    X, target = data()
    family = NormalObservations(lower=-0.5, upper=1.7)
    model = GAMLSS(
        formula={"mean": "~s(x,k=4)", "scale": "~1"},
        family=family,
        optimize_smoothing=True,
        smoothing_optimizer="bfgs",
        irls_tol=1e-9,
    ).fit(X, target)
    gam = model.gam_
    rho = np.log(gam.smoothing_params)
    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient,
        approx_derivative(
            lambda r: np.array([criterion_value(gam, gam.y_, r, method="reml")]), rho
        ).ravel(),
        atol=2e-5,
        rtol=2e-5,
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda r: criterion_gradient(gam, gam.y_, r, method="reml"), rho
        ).reshape(hessian.shape),
        atol=2e-5,
        rtol=2e-5,
    )
    neural = SemiStructuredLSS(
        objective=ObjectiveConfig(
            family="normal_observation",
            distributional_kwargs={"lower": -0.5, "upper": 1.7},
        ),
        architecture_config=DefaultSemiStructuredConfig(
            structured_formula="~x", hidden_dims=(3,), representation_dim=2
        ),
        training=SemiStructuredTrainingConfig(epochs=3, random_state=2),
    ).fit(
        X.iloc[:120],
        target.subset(slice(0, 120)),
        X_val=X.iloc[120:],
        y_val=target.subset(slice(120, None)),
    )
    law = neural.predict_distribution(X)
    assert np.isfinite(observation_log_prob(law, target)).all()
    assert np.isfinite(neural.score(X, target))
    assert neural.predict_components(X).link.shape == (len(X), 2)
    restored = neural.load_model(neural.save_model(tmp_path / "neural-normal.nampy"))
    assert_allclose(
        restored.predict_distribution(X).cdf([0.4] * len(X)),
        law.cdf([0.4] * len(X)),
        atol=0,
        rtol=0,
    )


def test_fitted_normal_observation_endpoint_matches_independent_r_capture():
    from pathlib import Path

    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / "normal_observations_input.csv")
    expected = pd.read_csv(root / "normal_observations.csv").iloc[0]
    target = ObservationTarget(source.lower, source.upper)
    X = source[["x"]]
    model = GAMLSS(
        formula={"mean": "~x", "scale": "~1"},
        family=NormalObservations(lower=-0.5, upper=1.7),
        optimize_smoothing=False,
        irls_tol=1e-10,
    ).fit(X, target)
    prediction = model.predict_link(X)
    assert_allclose(
        prediction[:, 0],
        expected.intercept + expected.slope * X.x.to_numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    assert_allclose(prediction[:, 1], expected.log_scale, rtol=2e-6, atol=2e-6)
    assert_allclose(
        -observation_log_prob(model.predict_distribution(X), target).sum(),
        expected.nll,
        atol=2e-8,
    )
