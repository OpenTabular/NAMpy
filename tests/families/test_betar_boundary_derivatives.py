"""Mixed beta derivatives checked independently of upstream cancellation."""

import numpy as np
import pytest

from nampy.gam.families.betar import BetaRegressionFamily
from tests.reference_fixtures import load_reference, reference_key, save_reference

ORDERS = {
    "Dmuth": (1, 1),
    "Dmu2th": (2, 1),
    "Dmuth2": (1, 2),
    "Dmu2th2": (2, 2),
    "Dmu3th": (3, 1),
}
EPS = np.finfo(float).eps * 100
MU = np.array([EPS, 1e-10, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-10, 1 - EPS])
Y = np.resize([1e-12, 0.2, 0.8, 1 - 1e-12], len(MU))
WEIGHTS = np.resize([0.25, 2.0, 1.0, 0.0], len(MU))


def high_precision_reference(theta):
    payload = {
        "theta": theta,
        "mu": MU.tolist(),
        "y": Y.tolist(),
        "weights": WEIGHTS.tolist(),
        "orders": ORDERS,
        "dps": 100,
    }
    key = reference_key("betar-logdensity-mpmath-diff-v1", payload)
    expected = load_reference("betar_high_precision", key)
    if expected is None:
        import mpmath as mp

        with mp.workdps(payload["dps"]):
            values = {name: [] for name in ORDERS}
            for y, mu, weight in zip(Y, MU, WEIGHTS, strict=True):
                y, mu, weight = (mp.mpf(float(x)) for x in (y, mu, weight))

                def deviance(m, log_theta, y=y, weight=weight):
                    t = mp.exp(log_theta)
                    # The saturated term is independent of m, so all mixed
                    # derivatives here equal those of the full deviance.
                    log_density = (
                        mp.loggamma(t)
                        - mp.loggamma(m * t)
                        - mp.loggamma((1 - m) * t)
                        + (m * t - 1) * mp.log(y)
                        + ((1 - m) * t - 1) * mp.log1p(-y)
                    )
                    return -2 * weight * log_density

                for name, order in ORDERS.items():
                    values[name].append(
                        float(mp.diff(deviance, (mu, mp.log(theta)), order))
                    )
        expected = {"mpmath_version": mp.__version__, "values": values}
        save_reference("betar_high_precision", key, expected)
    return expected["values"]


@pytest.mark.parametrize("theta", [0.001, 0.1, 10.0, 1e5])
def test_mixed_beta_derivatives_remain_accurate_near_mean_boundaries(theta):
    expected = high_precision_reference(theta)
    family = BetaRegressionFamily(theta=theta)
    actual = family.Dd(Y, MU, wt=WEIGHTS, level=2)
    for name, values in expected.items():
        np.testing.assert_allclose(
            actual[name], values, rtol=2e-11, atol=2e-10, err_msg=name
        )
    np.testing.assert_array_equal(actual["EDmu2th"], actual["Dmu2th"])
    first_order = family.Dd(Y, MU, wt=WEIGHTS, level=1)
    for name in ("Dmuth", "Dmu2th", "EDmu2th"):
        np.testing.assert_array_equal(first_order[name], actual[name])
