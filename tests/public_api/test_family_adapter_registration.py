"""Family declarations own public names and explicit probability adaptation."""

import numpy as np
import pytest

from nampy import GAMLSS
from nampy.gam.families.registry import GENERAL_FAMILIES, family_spec_for_instance
from nampy.gam.predict.distribution_adapter import distribution_prediction


@pytest.mark.parametrize(
    "name,source",
    [
        ("normal", "gaulss"),
        ("gamma", "gammals"),
        ("studentt", "student_lss"),
        ("beta", "beta_lss"),
        ("negativebinom", "nb_lss"),
        ("bccg", "bccg_lss"),
        ("bct", "bct_lss"),
        ("bcpe", "bcpe_lss"),
        ("betabinomial", "betabinomial_lss"),
        ("beinf", "beinf_lss"),
        ("zaga", "zaga_lss"),
        ("normal_observation", "normal_observation_lss"),
        ("lognormal_observation", "lognormal_observation_lss"),
        ("weibull_observation", "weibull_observation_lss"),
    ],
)
def test_gamlss_ordinary_names_resolve_to_source_specific_families(name, source):
    ordinary = GAMLSS(family=name)._resolved_family()
    explicit = GAMLSS(family=source)._resolved_family()
    assert type(ordinary) is type(explicit)
    assert family_spec_for_instance(ordinary) is GENERAL_FAMILIES[source]


def test_gamma_adapter_preserves_location_spread_parameterization():
    family = GAMLSS(family={"name": "gamma", "b": -5})._resolved_family()
    prediction = distribution_prediction(family, [[2.0, 0.5]])
    np.testing.assert_allclose(prediction.mean(), [2.0])
    np.testing.assert_allclose(prediction.variance(), [1.0])
    np.testing.assert_allclose(prediction.parameters["shape"], [4.0])
    np.testing.assert_allclose(prediction.parameters["rate"], [2.0])


def test_custom_statistical_subclasses_do_not_inherit_probability_claims():
    base = type(GAMLSS(family="normal")._resolved_family())

    class DifferentDensity(base):
        pass

    with pytest.raises(NotImplementedError, match="DifferentDensity"):
        distribution_prediction(DifferentDensity(), [[0.0, 1.0]])


def test_custom_neural_subclasses_do_not_inherit_probability_claims():
    from nampy.neural.families import NormalDistribution
    from nampy.neural.families.distribution_adapter import distribution_prediction

    class DifferentDensity(NormalDistribution):
        pass

    with pytest.raises(NotImplementedError, match="DifferentDensity"):
        distribution_prediction(DifferentDensity(), [[0.0, 1.0]])
