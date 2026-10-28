"""Tweedie null fitting must terminate at floating-point resolution."""

from importlib import import_module

import numpy as np

twlss_module = import_module("nampy.gam.families.gamlss.twlss")


def test_null_fit_compares_consistent_likelihood_reductions(monkeypatch):
    density = np.zeros((18, 10))
    density[:, 0] = [-1e16, *([-1.0] * 16), 1e16]
    density[:, [2, 4, 7]] = -1.0
    density[:, 3] = 1.0
    # Reproduce the NumPy column-sum/axis-sum discrepancy that stalled CI.
    assert np.sum(density[:, 0]) < np.sum(density, axis=0)[0]
    calls = 0

    def likelihood(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 64:
            raise AssertionError(
                "Null-fit line search repeatedly rejected equal likelihoods"
            )
        return density.copy()

    monkeypatch.setattr(twlss_module, "ldTweedie", likelihood)
    result = twlss_module._tw_null_fit(np.ones(18))
    assert np.isfinite(result).all()
    assert calls <= 51


def test_null_fit_stops_when_halving_cannot_change_parameters(monkeypatch):
    calls = 0

    def likelihood(y, mu, *, theta, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 64:
            raise AssertionError(
                "Null fit continued after the step lost numerical resolution"
            )
        density = np.zeros((len(y), 10))
        density[:, 0] = -((theta - 1.0) ** 2)
        density[:, [2, 4, 7]] = -1.0
        # A residual gradient at the exact maximum models cancellation in
        # the analytic derivatives without allowing a further uphill step.
        density[:, 3] = 1.0
        return density

    monkeypatch.setattr(twlss_module, "ldTweedie", likelihood)
    result = twlss_module._tw_null_fit(np.ones(18))
    np.testing.assert_array_equal(result[[0, 2]], [1.0, 1.0])
    assert calls < 64
