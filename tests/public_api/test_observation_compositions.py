"""Shared censoring/truncation semantics independent of either fitting engine."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from nampy.distributions import (
    DistributionPrediction,
    GammaLaw,
    NormalLaw,
    ObservationTarget,
    TruncatedLaw,
    observation_log_prob,
)


def test_interval_contract_subset_encoding_readonly_and_persistence():
    target = ObservationTarget(
        [0.2, -np.inf, 0.5, 0.3, -np.inf], [0.2, 0.1, np.inf, 0.8, np.inf]
    )
    restored = ObservationTarget.from_encoded(target.encoded)
    assert_array_equal(restored.lower, target.lower)
    assert_array_equal(restored.upper, target.upper)
    assert_array_equal(target.subset([2, 0, 2]).encoded, target.encoded[[2, 0, 2]])
    assert not pickle.loads(pickle.dumps(target)).lower.flags.writeable
    with pytest.raises(ValueError):
        ObservationTarget([1], [0])


def test_shared_normal_truncation_probabilities_moments_draws_and_tails():
    parameters = {"loc": np.array([0.0, 0.4]), "scale": np.array([1.0, 0.8])}
    prediction = DistributionPrediction(
        TruncatedLaw(NormalLaw(), -0.5, 1.7), parameters
    )
    source = stats.truncnorm(
        (-0.5 - parameters["loc"]) / parameters["scale"],
        (1.7 - parameters["loc"]) / parameters["scale"],
        loc=parameters["loc"],
        scale=parameters["scale"],
    )
    y = np.array([0.2, 0.7])
    assert_allclose(prediction.log_prob(y), source.logpdf(y), atol=1e-14)
    assert_allclose(prediction.cdf(y), source.cdf(y), atol=1e-14)
    assert_allclose(prediction.survival(y), source.sf(y), atol=1e-14)
    assert_allclose(prediction.mean(), source.mean(), atol=1e-14)
    assert_allclose(prediction.variance(), source.var(), atol=1e-14)
    assert_allclose(
        prediction.quantile([0, 0.2, 0.8, 1]),
        source.ppf(np.array([0, 0.2, 0.8, 1])[:, None]).T,
        atol=1e-14,
    )
    draws = prediction.sample(10, random_state=2)
    assert_array_equal(draws, prediction.sample(10, random_state=2))
    assert np.all((draws > -0.5) & (draws < 1.7))
    far = DistributionPrediction(
        TruncatedLaw(NormalLaw(), 35.0, 36.0),
        {"loc": np.array([0.0]), "scale": np.array([1.0])},
    )
    assert np.isfinite(far.log_prob([35.2])).all()
    assert_allclose(
        far.quantile([0.2, 0.8]),
        stats.truncnorm(35, 36).ppf([0.2, 0.8])[None, :],
        atol=1e-13,
    )


def test_censoring_likelihood_is_density_or_interval_mass_after_truncation():
    law = DistributionPrediction(
        TruncatedLaw(NormalLaw(), -0.5, 1.7), {"loc": np.zeros(5), "scale": np.ones(5)}
    )
    target = ObservationTarget(
        [0.2, -np.inf, 0.5, 0.3, -np.inf], [0.2, 0.1, np.inf, 0.8, np.inf]
    )
    source = stats.truncnorm(-0.5, 1.7)
    expected = [
        source.logpdf(0.2),
        source.logcdf(0.1),
        source.logsf(0.5),
        np.log(source.cdf(0.8) - source.cdf(0.3)),
        0.0,
    ]
    assert_allclose(observation_log_prob(law, target), expected, atol=1e-14)


def test_generic_gamma_probability_composition_earns_no_moment_capability():
    law = DistributionPrediction(
        TruncatedLaw(GammaLaw(), 0.4, 3.0),
        {"shape": np.array([2.0]), "rate": np.array([1.2])},
    )
    source = stats.gamma(2, scale=1 / 1.2)
    mass = source.cdf(3) - source.cdf(0.4)
    assert_allclose(law.log_prob([1.0]), source.logpdf(1) - np.log(mass), atol=1e-14)
    assert_allclose(
        law.quantile([0.2, 0.8]),
        source.ppf(source.cdf(0.4) + np.array([0.2, 0.8]) * mass)[None, :],
        atol=1e-14,
    )
    with pytest.raises(NotImplementedError):
        law.mean()


def test_generic_truncation_quantile_endpoints_respect_base_support():
    law = DistributionPrediction(
        TruncatedLaw(GammaLaw(), -1.0, np.inf),
        {"shape": np.array([2.0]), "rate": np.array([1.0])},
    )
    assert_allclose(law.quantile([0.0, 1.0]), [[0.0, np.inf]], atol=0, rtol=0)
