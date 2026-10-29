"""Independent positive-law censoring/truncation and analytic derivative checks."""

from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

from nampy import GAMLSS
from nampy.distributions import ObservationTarget, observation_log_prob
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.observations import (
    LogNormalObservations,
    WeibullObservations,
)
from nampy.neural.families.observations import (
    LogNormalObservationsDistribution,
    WeibullObservationsDistribution,
)


@pytest.mark.parametrize(
    "family_type,objective_type,point",
    [
        (LogNormalObservations, LogNormalObservationsDistribution, [0.2, -0.4]),
        (WeibullObservations, WeibullObservationsDistribution, [0.2, 0.4]),
    ],
)
def test_positive_observation_derivatives_through_four_and_probability(
    family_type, objective_type, point
):
    target = ObservationTarget(
        [0.6, -np.inf, 1.5, 0.7, -np.inf],
        [0.6, 0.5, np.inf, 1.1, np.inf],
    )
    family = family_type(lower=0.2, upper=3.0)
    encoded = target.encoded
    family.bind_fit_context(
        {"observation_upper": encoded[:, 1], "observation_code": encoded[:, 2]},
        len(target),
    )
    predictors = np.tile(point, (len(target), 1))
    variables = [Taylor(predictors[:, i], 2, 4, i) for i in range(2)]
    jet = family._log_density(encoded[:, 0], *variables)
    objective = objective_type(lower=0.2, upper=3.0)
    raw = torch.tensor(predictors, dtype=torch.float64, requires_grad=True)
    loss = objective.compute_loss(raw, torch.tensor(encoded), reduction="none")
    prediction = objective.distribution_prediction(objective(raw).detach().numpy())
    assert_allclose(jet.value, -loss.detach(), atol=1e-13)
    assert_allclose(jet.value, observation_log_prob(prediction, target), atol=1e-13)
    assert torch.autograd.gradcheck(
        lambda value: objective.compute_loss(value, torch.tensor(encoded)), (raw,)
    )

    def log_likelihood(b):
        return -objective.compute_loss(
            b.expand(len(target), 2), torch.tensor(encoded), reduction="sum"
        )

    derivative = log_likelihood
    for order in range(1, 5):
        derivative = torch.func.jacrev(derivative)
        expected = derivative(torch.tensor(point, dtype=torch.float64)).numpy()
        for indices in np.ndindex(*(2,) * order):
            multi = tuple(indices.count(i) for i in range(2))
            actual = (
                jet.coefficients[multi].sum()
                * factorial(multi[0])
                * factorial(multi[1])
            )
            assert_allclose(actual, expected[indices], rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("family", ["lognormal", "weibull"])
def test_positive_observation_fit_matches_independent_r(family, tmp_path):
    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / f"{family}_observations_input.csv")
    expected = pd.read_csv(root / f"{family}_observations.csv").iloc[0]
    target = ObservationTarget(source.lower, source.upper)
    X = source[["x"]]
    names = ("loc", "scale") if family == "lognormal" else ("scale", "shape")
    model = GAMLSS(
        formula={names[0]: "~x", names[1]: "~1"},
        family=(
            LogNormalObservations if family == "lognormal" else WeibullObservations
        )(lower=0.2, upper=3.0),
        optimize_smoothing=False,
        irls_tol=1e-10,
    ).fit(X, target)
    raw = model.predict(X, raw=True)
    assert_allclose(
        raw[:, 0],
        expected.intercept + expected.slope * X.x.to_numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    assert_allclose(raw[:, 1], expected.second, rtol=2e-6, atol=2e-6)
    prediction = model.predict_distribution(X)
    assert_allclose(
        -observation_log_prob(prediction, target).sum(), expected.nll, atol=2e-8
    )
    assert np.isfinite(model.predict_point(X)).all()
    restored = model.load_model(model.save_model(tmp_path / "observation.nampy"))
    assert_allclose(
        restored.predict_distribution(X).cdf(np.ones(len(X))),
        prediction.cdf(np.ones(len(X))),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("family", ["lognormal", "weibull"])
def test_positive_observation_neural_fit_and_conditional_summary(family):
    from nampy import GradientTrainingConfig, LinRegLSS, ObjectiveConfig

    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / f"{family}_observations_input.csv")
    target = ObservationTarget(source.lower, source.upper)
    model = LinRegLSS(
        objective=ObjectiveConfig(
            family=f"{family}_observation",
            distributional_kwargs={"lower": 0.2, "upper": 3.0},
        ),
        training=GradientTrainingConfig(
            max_epochs=1, random_state=2, trainer_kwargs={"enable_progress_bar": False}
        ),
    ).fit(source[["x"]], target)
    prediction = model.predict_distribution(source[["x"]])
    assert np.isfinite(observation_log_prob(prediction, target)).all()
    assert_allclose(
        model.predict_point(source[["x"]]), prediction.quantile(0.5)[:, 0], rtol=2e-7
    )
    assert model.predict_components(source[["x"]]).link.shape == (len(source), 2)


@pytest.mark.parametrize("family_type", [LogNormalObservations, WeibullObservations])
def test_positive_observation_automatic_smoothing(family_type):
    from nampy.gam.fit.selection.criteria import criterion_gradient, criterion_hessian

    root = Path(__file__).parents[1] / "reference_fixtures"
    name = "lognormal" if family_type is LogNormalObservations else "weibull"
    source = pd.read_csv(root / f"{name}_observations_input.csv")
    target = ObservationTarget(source.lower, source.upper)
    family = family_type(lower=0.2, upper=3.0)
    first, second = family.parameter_names
    model = GAMLSS(
        formula={first: "~s(x,k=4)", second: "~1"},
        family=family,
        smoothing_optimizer="bfgs",
        irls_tol=1e-9,
    ).fit(source[["x"]], target)
    rho = np.log(model.gam_.smoothing_params)
    assert np.isfinite(
        criterion_gradient(model.gam_, model.gam_.y_, rho, method="reml")
    ).all()
    assert np.isfinite(
        criterion_hessian(model.gam_, model.gam_.y_, rho, method="reml")
    ).all()
