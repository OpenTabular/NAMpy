"""Independent fourth-order likelihood validation before enabling REML."""

from itertools import combinations_with_replacement

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from nampy.gam.families.gamlss._taylor import Taylor, _normal_mills_tail_derivatives
from nampy.gam.families.gamlss.censored import CensoredLogNormal, CensoredWeibull
from nampy.survival import SurvivalTarget
from tests.survival.test_censored_derivatives import reference_loglik


@pytest.mark.parametrize(
    "z",
    [
        -40.0,
        -10.0,
        -3.0,
        0.0,
        3.0,
        7.0,
        7.999999,
        8.0,
        8.000001,
        10.0,
        40.0,
        80.0,
        200.0,
    ],
)
def test_log_survival_fourth_derivatives_against_high_precision(z):
    mp = pytest.importorskip("mpmath")
    value = Taylor(np.array([z]), 1, 4, 0).log_normal_survival()
    with mp.workdps(90):

        def log_survival(x):
            return mp.log(mp.erfc(x / mp.sqrt(2)) / 2)

        for degree in (1, 2, 3, 4):
            expected = float(mp.diff(log_survival, mp.mpf(z), degree))
            assert_allclose(
                value.packed_derivatives(degree)[0, 0], expected, atol=5e-12, rtol=2e-8
            )
    previous = Taylor(np.array([z]), 1, 2, 0).log_normal_survival()
    assert_array_equal(value.value, previous.value)
    assert_array_equal(value.packed_derivatives(1), previous.packed_derivatives(1))
    assert_array_equal(value.packed_derivatives(2), previous.packed_derivatives(2))
    if z >= 8:
        for actual, deeper in zip(
            _normal_mills_tail_derivatives(np.array([z])),
            _normal_mills_tail_derivatives(np.array([z]), depth=128),
            strict=True,
        ):
            assert_allclose(actual, deeper, rtol=2e-14, atol=0)


@pytest.mark.parametrize(
    "name,family_class",
    [("weibull", CensoredWeibull), ("lognormal", CensoredLogNormal)],
)
@pytest.mark.parametrize("event", [False, True])
@pytest.mark.parametrize("entry", [0.0, 0.4, 1.199])
def test_full_mixed_likelihood_tensor_with_entry(name, family_class, event, entry):
    target = SurvivalTarget([1.2], [event], [entry])
    family = family_class()
    family.bind_fit_context({"event": target.event, "entry": target.entry}, 1)
    eta = np.array([0.2, -0.3])
    variables = [Taylor(np.array([eta[k]]), 2, 4, k) for k in range(2)]
    value = family._log_density(target.time, *variables)

    def likelihood(z):
        return reference_loglik(name, z[None, :], target)[0]

    tensor = torch.tensor(eta, dtype=torch.float64)
    function = likelihood
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        derivative = function(tensor).numpy()
        packed = [
            derivative[index]
            for index in combinations_with_replacement(range(2), degree)
        ]
        assert_allclose(
            value.packed_derivatives(degree)[0], packed, atol=2e-9, rtol=2e-9
        )


@pytest.mark.parametrize("loc", [-39.0, 36.0])
@pytest.mark.parametrize("event", [False, True])
def test_lognormal_mixed_tail_derivatives_against_high_precision(loc, event):
    mp = pytest.importorskip("mpmath")
    target = SurvivalTarget([np.exp(1.0)], [event], [np.exp(0.5)])
    family = CensoredLogNormal()
    family.bind_fit_context({"event": target.event, "entry": target.entry}, 1)
    variables = [Taylor(np.array([loc]), 2, 4, 0), Taylor(np.array([0.0]), 2, 4, 1)]
    value = family._log_density(target.time, *variables)
    with mp.workdps(90):
        log_t, log_a = mp.log(float(target.time[0])), mp.log(float(target.entry[0]))

        def likelihood(mu, log_sd):
            z = (log_t - mu) * mp.exp(-log_sd)
            za = (log_a - mu) * mp.exp(-log_sd)
            observed = (
                -mp.log(2 * mp.pi) / 2 - log_sd - z * z / 2 - log_t
                if event
                else mp.log(mp.erfc(z / mp.sqrt(2)) / 2)
            )
            return observed - mp.log(mp.erfc(za / mp.sqrt(2)) / 2)

        for degree in range(1, 5):
            expected = []
            for combination in combinations_with_replacement(range(2), degree):
                counts = tuple(combination.count(k) for k in range(2))
                expected.append(
                    float(mp.diff(likelihood, (mp.mpf(loc), mp.mpf(0)), counts))
                )
            assert_allclose(
                value.packed_derivatives(degree)[0], expected, atol=2e-9, rtol=2e-9
            )
