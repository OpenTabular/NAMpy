"""Rare-event derivatives checked independently of double-precision mgcv loss."""

import math

import numpy as np
import pytest

from nampy.gam.families.cpois import CensoredPoissonFamily
from nampy.gam.response import CensoredResponse
from tests.reference_fixtures import load_reference, reference_key, save_reference

CASES = [
    (1000.0, "Inf", 1.0),
    (1000.0, 1001.0, 1.0),
    (200.0, 210.0, 1.0),
    (5.0, "-Inf", 1000.0),
    (0.0, -1.0, 1000.0),
    (1001.0, 1001.0, 1.0),
    (1000.2, 1001.8, 1.0),
    (1000.0, 1001.0, 0.1),
    (210.0, 200.0, 1.0),
    (50.0, "Inf", 10.0),
    (50.0, 60.0, 10.0),
]


def high_precision_reference():
    payload = {"cases": CASES, "dps": 100, "orders": [1, 2, 3, 4]}
    key = reference_key("cpois-poisson-event-derivatives-mpmath-v1", payload)
    result = load_reference("cpois_tail_high_precision", key)
    if result is None:
        import mpmath as mp

        rows = []
        with mp.workdps(payload["dps"]):
            for value, bound, mean in CASES:
                bound = float(bound)

                def deviance(mu, value=value, bound=bound):
                    if value == bound:
                        probability = (
                            mp.exp(-mu) * mu ** int(value) / mp.factorial(int(value))
                        )
                    elif bound == -math.inf:
                        upper = math.floor(value + 1e-7)
                        probability = mp.gammainc(
                            upper + 1, mu, mp.inf, regularized=True
                        )
                    elif bound == math.inf:
                        lower = math.floor(value + 1e-7) + 1
                        probability = mp.gammainc(lower, 0, mu, regularized=True)
                    else:
                        lower = max(0, math.floor(min(value, bound) + 1e-7) + 1)
                        upper = math.floor(max(value, bound) + 1e-7)
                        probability = mp.fsum(
                            mp.exp(-mu) * mu**k / mp.factorial(k)
                            for k in range(lower, upper + 1)
                        )
                    return -2 * mp.log(probability)

                rows.append(
                    [
                        float(mp.diff(deviance, mp.mpf(mean), order))
                        for order in payload["orders"]
                    ]
                )
        result = {"mpmath_version": mp.__version__, "values": rows}
        save_reference("cpois_tail_high_precision", key, result)
    return np.asarray(result["values"])


@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_rare_poisson_event_derivatives_match_high_precision(link):
    expected = high_precision_reference()
    y = CensoredResponse([c[0] for c in CASES], [float(c[1]) for c in CASES])
    mu = np.asarray([c[2] for c in CASES])
    family = CensoredPoissonFamily(link)
    actual = family.Dd(y, mu, level=2)
    for column, name in enumerate(("Dmu", "Dmu2", "Dmu3", "Dmu4")):
        np.testing.assert_allclose(
            actual[name], expected[:, column], rtol=2e-12, atol=2e-25, err_msg=name
        )
    np.testing.assert_array_equal(actual["EDmu2"], actual["Dmu2"])
    assert actual["Dmu4"][1] == 12012.0
    for level in (0, 1):
        for name, values in family.Dd(y, mu, level=level).items():
            np.testing.assert_array_equal(values, actual[name])


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_tail_derivatives_ignore_one_ulp_likelihood_ratio_rounding(
    monkeypatch, direction
):
    from nampy.gam.families import cpois

    response = CensoredResponse([c[0] for c in CASES], [float(c[1]) for c in CASES])
    means = np.asarray([c[2] for c in CASES])
    family = CensoredPoissonFamily()
    baseline = family.Dd(response, means, level=2)
    original = cpois.exp

    def rounded(values):
        result = original(values)
        return np.where(
            np.isfinite(result) & (result > 0), np.nextafter(result, direction), result
        )

    # Valid last-bit libm differences must not be amplified into the result.
    monkeypatch.setattr(cpois, "exp", rounded)
    for name, values in family.Dd(response, means, level=2).items():
        np.testing.assert_array_equal(values, baseline[name])
