"""Backend-owned fit status and real held-out statistical model comparison."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from nampy import GAMLSS
from nampy.distributions import DistributionCandidate, compare_distributions
from nampy.gam.diagnostics.fit_status import distribution_fit_status
from nampy.gam.model_state import _fit_result


def _data():
    rng = np.random.default_rng(112)
    train = pd.DataFrame({"x": rng.uniform(-1, 1, 90)})
    valid = pd.DataFrame({"x": rng.uniform(-1, 1, 24)})
    y = rng.lognormal(0.2 + 0.3 * train.x, 0.4)
    z = rng.lognormal(0.2 + 0.3 * valid.x, 0.4)
    return train, y, valid, z


def _lognormal(**kwargs):
    return GAMLSS(
        family="lognormal_lss",
        formula={"loc": "y ~ x", "scale": "~ 1"},
        optimize_smoothing=False,
        **kwargs,
    )


def test_real_statistical_candidates_match_public_normalized_scores_and_preserve_templates():
    first = _lognormal()
    second = GAMLSS(
        family="weibull_lss",
        formula={"scale": "y ~ x", "shape": "~ 1"},
        optimize_smoothing=False,
    )
    weights = np.tile([0.0, 1.0, 2.0], 30)
    validation = np.tile([0.0, 1.0, 2.0], 8)
    X, y, V, z = _data()
    result = compare_distributions(
        {"lognormal": first, "weibull": second},
        X,
        y,
        V,
        z,
        sample_weight=weights,
        validation_weight=validation,
    )
    assert len(result.ranking) == 2
    for row in result.results:
        assert row.fit_status.backend == "gam" and row.fit_status.state == "converged"
        assert row.calibration.n_used.tolist() == [16, 16, 16]
        assert_allclose(
            row.mean_log_prob,
            row.estimator.score(V, z, sample_weight=validation),
            atol=2e-14,
        )
        assert row.estimator.distribution_fit_status() == row.fit_status
    assert not hasattr(first, "gam_") and not hasattr(second, "gam_")


def test_statistical_fit_status_reads_both_coefficient_and_outer_failures(monkeypatch):
    X, y, V, z = _data()
    model = _lognormal().fit(X, y)
    assert model.distribution_fit_status().state == "converged"
    gam = model.gam_
    core = _fit_result(gam)
    monkeypatch.setattr(
        "nampy.gam.diagnostics.fit_status._fit_result",
        lambda model: replace(
            core, converged=False, failure_reason="test coefficient failure"
        ),
    )
    assert model.distribution_fit_status().state == "nonconverged"
    assert "coefficient failure" in model.distribution_fit_status().message
    monkeypatch.setattr(
        "nampy.gam.diagnostics.fit_status._fit_result", lambda model: core
    )
    gam._optim_result = SimpleNamespace(success=False, message="test outer failure")
    assert model.distribution_fit_status().state == "nonconverged"
    gam._optim_result = SimpleNamespace(message="no reported success")
    assert model.distribution_fit_status().state == "unknown"
    assert distribution_fit_status(None).state == "unknown"
    assert GAMLSS().distribution_fit_status().state == "unknown"


def test_trial_context_and_prediction_offsets_remain_explicit_and_not_fit_weights():
    rng = np.random.default_rng(22)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 120)})
    trials = rng.integers(3, 15, len(X))
    y = rng.binomial(trials, rng.beta(2, 3, len(X)))
    V = X.iloc[:20].copy()
    valid_trials = np.full(len(V), 9)
    z = rng.binomial(valid_trials, 0.4)
    kwargs = {
        "family": "betabinomial_lss",
        "formula": {"mu": "y ~ x", "kappa": "~ 1"},
        "optimize_smoothing": False,
    }
    candidate = DistributionCandidate(
        GAMLSS(**kwargs),
        fit_kwargs={"context": {"trials": trials}},
        predict_kwargs={
            "context": {"trials": valid_trials},
            "offset": [np.zeros(len(V)), np.zeros(len(V))],
        },
    )
    missing_context = DistributionCandidate(
        GAMLSS(**kwargs), fit_kwargs={"context": {"trials": trials}}
    )
    result = compare_distributions(
        {"valid": candidate, "missing_context": missing_context},
        X,
        y,
        V,
        z,
        sample_weight=np.ones(len(X)),
    )
    assert result.ranking == ("valid",)
    assert result.results[1].status == "evaluation_error"
    assert "context" in result.results[1].failure_reason
    direct = result.models["valid"].predict_distribution(
        V, context={"trials": valid_trials}, offset=[np.zeros(len(V)), np.zeros(len(V))]
    )
    assert_allclose(result.results[0].mean_log_prob, direct.log_prob(z).mean())


def test_comparison_preserves_backend_status_instead_of_inferring_from_budget():
    X, y, V, z = _data()
    limited = GAMLSS(
        family="weibull_lss",
        formula={"scale": "y ~ x", "shape": "~ x"},
        optimize_smoothing=False,
        max_irls_iter=1,
    )
    result = compare_distributions(
        {"limited": limited, "good": _lognormal()}, X, y, V, z
    )
    # gam.fit5's maxit branch can report convergence at the requested budget.
    # Comparison retains that source status without another stopping rule.
    assert _fit_result(result.models["limited"].gam_).converged
    assert result.results[0].fit_status.state == "converged"
    assert result.results[0].eligible
