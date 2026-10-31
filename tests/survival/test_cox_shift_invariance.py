"""Cox risk-set algebra and baseline inference ignore stratum constants."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy import CoxGAM
from nampy.gam.losses.cox import cox_higher_derivatives, cox_partial_likelihood
from nampy.survival import SurvivalTarget


@pytest.mark.parametrize("shifts", [(400, 0), (1000, -1000), (-1000, 1000)])
def test_stratum_shift_preserves_likelihood_and_derivatives(shifts):
    matrix = np.array([[-1.0], [0.2], [1.0], [0.5], [-0.5], [0.3]])
    beta = np.array([0.4])
    time = np.array([1, 2, 3, 1, 2, 3.0])
    event = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
    strata = np.repeat([0, 1], 3)
    original = np.zeros(6)
    shifted = np.repeat(shifts, 3)
    for offset in (original, shifted):
        result = cox_partial_likelihood(beta, matrix, time, event, strata, offset)
        higher = cox_higher_derivatives(beta, matrix, time, event, strata, offset)
        if offset is original:
            reference, reference_higher = result, higher
        else:
            for actual, expected in zip(result, reference, strict=True):
                assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
            for actual, expected in zip(higher, reference_higher, strict=True):
                assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_wide_predictor_range_has_nonempty_normalized_risk_sets():
    # A single global exp(eta-max(eta)) would underflow the first risk set.
    matrix = np.array([[-1.0], [0.0], [1.0]])
    result = cox_partial_likelihood(
        np.array([0.0]), matrix, np.array([3.0, 2.0, 1.0]),
        np.ones(3, dtype=bool), np.zeros(3, dtype=int), np.array([-2000., 0., 2000.]),
    )
    for value in result:
        assert_allclose(value, np.zeros_like(value), atol=1e-12)


@pytest.mark.parametrize("shifts", [(1000, -1000), (-1000, 1000)])
def test_public_cox_predictions_uncertainty_and_diagnostics_are_shift_invariant(shifts):
    data = pd.read_csv(Path(__file__).parents[1] / "reference_fixtures/cox_fixed_input.csv")
    X, target = data[["x", "z"]], SurvivalTarget(data.time, data.event)
    strata = data.stratum.to_numpy()
    offset = data.offset.to_numpy()
    shift = np.where(strata == strata[0], shifts[0], shifts[1])
    models = [CoxGAM(formula="~x+z", tol=1e-8).fit(
        X, target, strata=strata, offset=known
    ) for known in (offset, offset + shift)]
    for method, kwargs in (
        ("predict", {"times": [0.5, 1, 2]}),
        ("predict_standard_error", {"times": [0.5, 1, 2]}),
        ("score", {"y": target}),
    ):
        values = [getattr(model, method)(X, strata=strata, offset=known, **kwargs)
                  for model, known in zip(models, (offset, offset + shift), strict=True)]
        assert_allclose(values[0], values[1], rtol=2e-7, atol=2e-7)
    diagnostics = [model.diagnostics(X, target, strata=strata, offset=known)
                   for model, known in zip(models, (offset, offset + shift), strict=True)]
    for field in ("schoenfeld", "martingale", "deviance", "cumulative_score"):
        assert_allclose(getattr(diagnostics[0], field), getattr(diagnostics[1], field),
                        rtol=2e-7, atol=2e-7, equal_nan=True)


def test_fitted_reference_balances_risk_and_inverse_risk_variance(monkeypatch):
    from nampy.gam.fit.solvers.cox import CoxSolution

    # Isolate baseline publication from optimization: this valid numerical
    # endpoint has finite hazards and variance in a balanced predictor frame.
    # Centering only at max(eta) would overflow its inverse squared risks.
    monkeypatch.setattr("nampy.models.cox.fit_cox", lambda *args, **kwargs:
                        CoxSolution(np.array([200.]), np.eye(1), -1., 1))
    X = pd.DataFrame({"x": [-1., 0., 1.]})
    target = SurvivalTarget([3., 2., 1.], [1, 1, 1])
    model = CoxGAM(formula="~x").fit(X, target)
    assert np.isfinite(model.baseline_[0][1]).all()
    assert np.isfinite(model.baseline_uncertainty_[0][1]).all()
    assert np.isfinite(model.predict_standard_error(X, times=[1., 2., 3.])).all()
