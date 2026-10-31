"""New censoring metadata is consumed by public statistical likelihood scoring."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy import GAMLSS
from nampy.distributions import ObservationTarget, observation_log_prob


@pytest.mark.parametrize("weighted", [False, True])
def test_observation_score_and_evaluate_use_new_target_bounds(weighted):
    rng = np.random.default_rng(41)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 80)})
    response = rng.normal(0.4 + 0.2 * X.x, 0.7)
    training = ObservationTarget(response, response)
    model = GAMLSS(formula=["~x", "~1"], family="normal_observation_lss").fit(X, training)
    # Same row count, different ordering and censoring. Training metadata must
    # never supply the new experiment's bounds implicitly.
    query = X.iloc[::-1].reset_index(drop=True)
    lower, upper = response[::-1].copy(), response[::-1].copy()
    lower[::4] = -np.inf
    upper[1::4] = np.inf
    upper[2::4] += 0.3
    target = ObservationTarget(lower, upper)
    weights = np.linspace(0.5, 2, len(query)) if weighted else None
    prediction = model.predict_distribution(query)
    expected = np.average(observation_log_prob(prediction, target), weights=weights)
    assert_allclose(model.score(query, target, sample_weight=weights), expected)
    assert_allclose(
        model.evaluate(query, target, sample_weight=weights)["Negative Log-Likelihood"],
        -expected,
    )
    exact = np.average(prediction.log_prob(response[::-1]), weights=weights)
    assert_allclose(model.score(query, response[::-1], sample_weight=weights), exact)
    assert not np.isclose(expected, exact)


def test_observation_scoring_requires_new_prediction_context_and_offsets():
    from nampy.gam.families.gamlss.observations import NormalObservations

    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    response = 0.2 + 0.1 * X.x.to_numpy() + 0.3 * np.sin(np.arange(60))
    target = ObservationTarget(response, response)
    offset = np.full((len(X), 2), 0.1)
    model = GAMLSS(formula=["~x", "~1"], family=NormalObservations()).fit(
        X, target, offset=offset
    )
    query, target = X.iloc[:7], ObservationTarget(response[:7] - 0.1, response[:7] + 0.1)
    prediction = model.predict_distribution(query, offset=offset[:7])
    expected = observation_log_prob(prediction, target).mean()
    assert_allclose(model.score(query, target, offset=offset[:7]), expected)
    assert_allclose(model.evaluate(query, target, offset=offset[:7])["Negative Log-Likelihood"], -expected)


def test_uninformative_observations_contribute_zero_to_held_out_weighted_scoring():
    from nampy.gam.families.gamlss.observations import NormalObservations

    rng = np.random.default_rng(82)
    train = pd.DataFrame({"x": np.linspace(-1, 1, 70)})
    response = rng.normal(0.2 + 0.4 * train.x, 0.45)
    model = GAMLSS(
        formula=["~x", "~1"], family=NormalObservations(lower=-2, upper=2)
    ).fit(train, ObservationTarget(response, response))
    query = pd.DataFrame({"x": [-0.85, -0.15, 0.37, 0.9, 1.1]})
    target = ObservationTarget(
        [-np.inf, -0.4, -np.inf, 0.0, -np.inf],
        [np.inf, np.inf, 0.3, 0.8, np.inf],
    )
    weights = np.array([2.0, 1.0, 3.0, 0.0, 4.0])
    contributions = observation_log_prob(model.predict_distribution(query), target)
    assert_allclose(contributions[[0, 4]], 0.0, atol=1e-15)
    assert np.isfinite(contributions).all()
    expected = weights @ contributions / weights.sum()
    assert_allclose(model.score(query, target, sample_weight=weights), expected)
    assert_allclose(
        model.evaluate(query, target, sample_weight=weights)["Negative Log-Likelihood"],
        -expected,
    )
    # Uninformative rows still belong in the requested weighted average.
    assert not np.isclose(expected, np.average(contributions[1:4], weights=weights[1:4]))
    uninformative = ObservationTarget(np.full(len(query), -np.inf), np.full(len(query), np.inf))
    assert_allclose(model.score(query, uninformative, sample_weight=weights), 0.0, atol=1e-15)
    assert_allclose(
        model.evaluate(query, uninformative, sample_weight=weights)["Negative Log-Likelihood"],
        0.0, atol=1e-15,
    )


def test_custom_evaluation_metrics_receive_offset_aware_held_out_parameters():
    rng = np.random.default_rng(93)
    train = pd.DataFrame({"x": np.linspace(-1, 1, 70)})
    training_offset = np.column_stack([0.4 * train.x, 0.2 * np.sin(train.x)])
    response = (
        0.3 + 0.8 * train.x.to_numpy() + training_offset[:, 0]
        + rng.normal(size=len(train)) * 0.5 * np.exp(training_offset[:, 1])
    )
    model = GAMLSS(formula=["~x", "~1"], family="normal").fit(
        train, response, offset=training_offset
    )
    query = pd.DataFrame({"x": [-0.9, -0.25, 0.35, 0.8, 1.1]})
    offset = np.array([[-1, 0.2], [-0.5, -0.1], [0.2, 0.4], [0.75, -0.2], [1.3, 0.1]])
    target = np.array([-0.3, 1.0, 0.2, 2.0, 1.5])
    expected = model.predict_parameters(query, offset=offset)
    assert not np.allclose(expected, model.predict_parameters(query, offset=np.zeros_like(offset)))
    received = []

    def mean_squared_error(y, parameters):
        assert_allclose(y, target)
        received.append(parameters.copy())
        return float(np.mean((y - parameters[:, 0]) ** 2))

    result = model.evaluate(
        query, target, metrics={"Mean squared error": mean_squared_error}, offset=offset
    )
    assert len(received) == 1
    assert_allclose(received[0], expected)
    assert_allclose(result["Mean squared error"], np.mean((target - expected[:, 0]) ** 2))
    weights = np.array([1.0, 0.5, 2.0, 0.0, 3.0])
    assert_allclose(
        model.evaluate(query, target, offset=offset, sample_weight=weights)["Negative Log-Likelihood"],
        -model.score(query, target, offset=offset, sample_weight=weights),
    )
