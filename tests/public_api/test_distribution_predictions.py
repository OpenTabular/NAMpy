"""Scientific contracts for backend-neutral conditional distributions."""

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from nampy.distributions.contracts import DistributionSpec
from nampy.distributions import DistributionPrediction, GammaLaw, NormalLaw


def test_normal_probability_identities_shapes_and_tail_operations():
    prediction = DistributionPrediction(
        NormalLaw(), {"loc": [-2.0, 3.0], "scale": [0.5, 2.0]}
    )
    assert_allclose(prediction.cdf([-2.0, 3.0]), 0.5)
    assert_allclose(
        prediction.log_prob([-2.0, 3.0]),
        -np.log(np.array([0.5, 2.0]) * np.sqrt(2 * np.pi)),
    )
    assert_allclose(prediction.mean(), [-2.0, 3.0])
    assert_allclose(prediction.variance(), [0.25, 4.0])
    assert_allclose(prediction.quantile(0.5), [[-2.0], [3.0]])
    assert prediction.quantile([]).shape == (2, 0)
    assert_array_equal(prediction.cdf_left(0), prediction.cdf(0))
    assert_array_equal(prediction.cdf([-np.inf, np.inf]), [0, 1])
    assert np.all(np.isneginf(prediction.quantile(0)))
    assert np.all(np.isposinf(prediction.quantile(1)))
    # Normal tails must not be computed as log(1 - cdf), which rounds to -inf.
    upper = prediction.parameters["loc"] + 40 * prediction.parameters["scale"]
    assert_array_equal(prediction.survival(upper), [0.0, 0.0])
    assert np.all(np.isfinite(prediction.log_survival(upper)))
    lower = prediction.parameters["loc"] - 40 * prediction.parameters["scale"]
    assert np.all(np.isfinite(prediction.log_cdf(lower)))
    probs = np.array([0.01, 0.25, 0.9])
    quantiles = prediction.quantile(probs)
    assert quantiles.shape == (2, 3)
    for j, prob in enumerate(probs):
        assert_allclose(prediction.cdf(quantiles[:, j]), prob)


def test_gamma_matches_exponential_and_erlang_identities():
    rate = np.array([2.0, 3.0])
    prediction = DistributionPrediction(GammaLaw(), {"shape": [1.0, 2.0], "rate": rate})
    y = np.array([0.4, 0.6])
    z = rate * y
    expected_cdf = [1 - np.exp(-z[0]), 1 - (1 + z[1]) * np.exp(-z[1])]
    expected_density = [rate[0] * np.exp(-z[0]), rate[1] ** 2 * y[1] * np.exp(-z[1])]
    assert_allclose(prediction.cdf(y), expected_cdf)
    assert_allclose(prediction.log_prob(y), np.log(expected_density))
    assert_allclose(prediction.survival(y), 1 - np.array(expected_cdf))
    assert_allclose(prediction.log_survival(y), np.log1p(-np.array(expected_cdf)))
    assert_allclose(prediction.log_cdf(y), np.log(expected_cdf))
    assert_allclose(prediction.mean(), np.array([1, 2]) / rate)
    assert_allclose(prediction.variance(), np.array([1, 2]) / rate**2)
    assert_array_equal(prediction.cdf(-1), [0, 0])
    assert np.all(np.isneginf(prediction.log_prob(-1)))
    assert_array_equal(prediction.quantile(0), [[0], [0]])
    assert_array_equal(prediction.cdf_left(0), [0, 0])
    assert_allclose(prediction.log_prob(0)[0], np.log(rate[0]))
    quantiles = prediction.quantile([0.05, 0.4, 0.95])
    for j, probability in enumerate([0.05, 0.4, 0.95]):
        assert_allclose(prediction.cdf(quantiles[:, j]), probability, atol=1e-13)


@pytest.mark.parametrize(
    "law,parameters,lower,upper",
    [
        (NormalLaw(), {"loc": [2.0], "scale": [0.7]}, -np.inf, np.inf),
        (GammaLaw(), {"shape": [0.6], "rate": [1.3]}, 0, np.inf),
    ],
)
def test_probability_density_normalizes(law, parameters, lower, upper):
    prediction = DistributionPrediction(law, parameters)
    integral, _ = quad(lambda y: np.exp(prediction.log_prob(y)[0]), lower, upper)
    assert integral == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    "law,parameters",
    [
        (NormalLaw(), {"loc": [-2.0, 1.0], "scale": [0.5, 2.0]}),
        (GammaLaw(), {"shape": [0.7, 3.0], "rate": [2.0, 1.0]}),
    ],
)
def test_sampling_is_reproducible_with_expected_moments(law, parameters):
    prediction = DistributionPrediction(law, parameters)
    draws = prediction.sample(30000, random_state=13)
    assert draws.shape == (30000, 2)
    assert_array_equal(draws, prediction.sample(30000, random_state=13))
    assert_array_equal(
        prediction.sample(3, random_state=np.random.default_rng(7)),
        prediction.sample(3, random_state=7),
    )
    assert_allclose(draws.mean(axis=0), prediction.mean(), atol=0.025)
    assert_allclose(draws.var(axis=0), prediction.variance(), rtol=0.03)
    assert prediction.sample(0).shape == (0, 2)


def test_snapshot_preserves_missing_rows_and_survives_pickle():
    loc = np.array([1.0, np.nan, 2.0])
    prediction = DistributionPrediction(
        NormalLaw(), {"loc": loc, "scale": [1.0, np.nan, 3.0]}
    )
    loc[0] = 99
    assert prediction.mean()[0] == 1
    assert prediction.n_rows == 3
    for result in [
        prediction.mean(),
        prediction.cdf(0),
        prediction.quantile([0.1, 0.9]),
        prediction.sample(4, random_state=3).T,
    ]:
        assert np.all(np.isnan(result[1]))
    with pytest.raises(ValueError):
        prediction.parameters["loc"][0] = 0
    with pytest.raises(TypeError):
        prediction.parameters["other"] = np.zeros(3)
    restored = pickle.loads(pickle.dumps(prediction))
    assert_array_equal(restored.mean(), prediction.mean())
    assert not restored.parameters["loc"].flags.writeable
    for parameters in [{"loc": [], "scale": []}, {"loc": [np.nan], "scale": [np.nan]}]:
        result = DistributionPrediction(NormalLaw(), parameters)
        assert result.quantile([0.2, 0.5]).shape == (len(parameters["loc"]), 2)
        assert result.sample(3).shape == (3, len(parameters["loc"]))


@pytest.mark.parametrize(
    "parameters",
    [
        {"loc": [0.0]},
        {"loc": [0.0], "scale": [1.0], "extra": [1.0]},
        {"loc": [[0.0]], "scale": [1.0]},
        {"loc": [0.0, 1.0], "scale": [1.0]},
        {"loc": [np.nan], "scale": [1.0]},
        {"loc": [np.inf], "scale": [1.0]},
        {"loc": [0.0], "scale": [0.0]},
        {"loc": [0.0], "scale": [-1.0]},
    ],
)
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        DistributionPrediction(NormalLaw(), parameters)


@pytest.mark.parametrize(
    "method,argument",
    [
        ("cdf", [[0.0]]),
        ("log_prob", [1.0, 2.0]),
        ("quantile", [np.nan]),
        ("quantile", [-0.1]),
        ("quantile", [1.1]),
        ("quantile", [[0.5]]),
        ("sample", -1),
        ("sample", 1.2),
        ("sample", True),
    ],
)
def test_ambiguous_shapes_and_invalid_arguments_are_rejected(method, argument):
    prediction = DistributionPrediction(NormalLaw(), {"loc": [0.0], "scale": [1.0]})
    with pytest.raises(ValueError):
        getattr(prediction, method)(argument)


@dataclass(frozen=True)
class _TwoPointLaw:
    """An external law exercises jumps without depending on built-in names."""

    spec = DistributionSpec(
        "two_point", ("p",), frozenset({"cdf", "cdf_left"}), "p_at_one_v1", "discrete"
    )

    def validate_parameters(self, parameters, *, context):
        if np.any((parameters["p"] < 0) | (parameters["p"] > 1)):
            raise ValueError("p must be in [0, 1].")

    def evaluate(self, operation, parameters, *, values=None, **kwargs):
        p = parameters["p"]
        if operation == "cdf":
            return np.where(values < 0, 0.0, np.where(values < 1, 1 - p, 1.0))
        return np.where(values <= 0, 0.0, np.where(values <= 1, 1 - p, 1.0))


def test_external_law_can_supply_left_limits_without_other_operations():
    prediction = DistributionPrediction(_TwoPointLaw(), {"p": [0.3, 0.8]})
    assert_array_equal(prediction.cdf([0.0, 1.0]), [0.7, 1.0])
    assert_allclose(prediction.cdf_left([0.0, 1.0]), [0.0, 0.2])
    with pytest.raises(
        NotImplementedError, match="two_point.*quantile.*Available operations"
    ):
        prediction.quantile(0.5)
    # Even an all-missing snapshot must reject unsupported operations.
    missing = DistributionPrediction(_TwoPointLaw(), {"p": [np.nan]})
    with pytest.raises(NotImplementedError, match="mean"):
        missing.mean()


@dataclass(frozen=True)
class _ContextLaw:
    """Two atoms at zero and a known row-specific upper endpoint."""

    spec = DistributionSpec(
        "context_law",
        ("p",),
        frozenset({"cdf", "cdf_left"}),
        "p_at_upper_v1",
        "discrete",
        context_names=("upper",),
    )

    def validate_parameters(self, parameters, *, context):
        if np.any(~np.isfinite(context["upper"])) or np.any(context["upper"] <= 0):
            raise ValueError("upper must be finite and positive.")

    def evaluate(self, operation, parameters, *, context, values=None, **kwargs):
        p = parameters["p"]
        upper = context["upper"]
        if operation == "cdf":
            return np.where(values < 0, 0.0, np.where(values < upper, 1 - p, 1.0))
        return np.where(values <= 0, 0.0, np.where(values <= upper, 1 - p, 1.0))


def test_row_context_remains_aligned_when_missing_rows_are_removed():
    upper = np.array([1.0, np.nan, 2.0])
    prediction = DistributionPrediction(
        _ContextLaw(), {"p": [0.3, np.nan, 0.8]}, {"upper": upper}
    )
    upper[2] = 99
    assert_allclose(prediction.cdf([1.0, 0.0, 2.0]), [1.0, np.nan, 1.0])
    assert_allclose(prediction.cdf_left([1.0, 0.0, 2.0]), [0.7, np.nan, 0.2])
    restored = pickle.loads(pickle.dumps(prediction))
    assert_array_equal(restored.context["upper"], [1.0, np.nan, 2.0])
    assert not restored.context["upper"].flags.writeable
    with pytest.raises(ValueError, match="context"):
        DistributionPrediction(_ContextLaw(), {"p": [0.5]})
    with pytest.raises(ValueError, match="shape"):
        DistributionPrediction(_ContextLaw(), {"p": [0.5]}, {"upper": [1.0, 2.0]})
    with pytest.raises(ValueError, match="positive"):
        DistributionPrediction(_ContextLaw(), {"p": [0.5]}, {"upper": [-1.0]})
