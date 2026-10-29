"""Both fitting engines consume exactly the shared diagnostic/evaluation API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.diagnostics import survival_residuals
from nampy.survival import (
    CensoringDistribution,
    SurvivalTarget,
    brier_score,
    concordance_index,
)
from tests.survival.test_censored_fitting import sample


@pytest.mark.parametrize("backend", ["statistical", "neural"])
@pytest.mark.parametrize("name", ["weibull", "lognormal", "piecewise_exponential"])
def test_fitted_diagnostics_rmst_and_evaluation_across_backends(backend, name, tmp_path):
    X, target, weights = sample("lognormal" if name == "lognormal" else "weibull")
    cuts = [0, .4, 1.2, 3] if name == "piecewise_exponential" else None
    offsets = (np.full(len(X), .03) if cuts else
               np.tile([.03, -.02], (len(X), 1)))
    if backend == "statistical":
        model = SurvivalGAM(likelihood=name, cuts=cuts, formula="~x", optimize_smoothing=False)
    else:
        model = LinRegSurvival(
            objective=SurvivalObjectiveConfig(likelihood=name, cuts=cuts),
            training=GradientTrainingConfig(max_epochs=1, batch_size=64,
                checkpoint_path=str(tmp_path), random_state=19,
                trainer_kwargs={"logger": False, "enable_progress_bar": False,
                                "enable_model_summary": False}),
        )
    model.fit(X, target, offset=offsets, sample_weight=weights)
    prediction = model.predict_distribution(X, offset=offsets)
    before = model.predict_link(X, offset=offsets)
    score_before = model.score(X, target, offset=offsets, sample_weight=weights)
    residuals = survival_residuals(prediction, target)
    assert_allclose(residuals.cox_snell,
                    prediction.cumulative_hazard_at(target.time, conditional_on=target.entry))
    remaining = prediction.rmst(2, conditional_on=target.entry)
    assert np.isfinite(remaining).all() and np.all(remaining >= 0)
    assert np.all(remaining <= 2-target.entry)
    # A separate no-entry evaluation cohort uses the same covariate rows.
    evaluation = SurvivalTarget(np.linspace(.15, 2.95, len(X)), np.arange(len(X)) % 3 != 0)
    censoring = CensoringDistribution.fit(SurvivalTarget([.1, .5, 4], [1, 0, 1]))
    result = brier_score(prediction, evaluation, [.2, .8, 1.5], censoring=censoring,
                         sample_weight=weights)
    assert np.isfinite(result.score).all() and result.integrated() >= 0
    risk = -prediction.rmst(2)
    concordance = concordance_index(evaluation, risk, higher_risk=True, sample_weight=weights)
    assert 0 <= concordance.concordance <= 1
    restored = type(model).load_model(model.save_model(tmp_path / "evaluation.nampy"))
    snapshot = restored.predict_distribution(X, offset=offsets)
    assert_allclose(snapshot.rmst(2, conditional_on=target.entry), remaining, atol=0, rtol=0)
    assert_array_equal(survival_residuals(snapshot, target).cox_snell, residuals.cox_snell)
    assert_array_equal(before, model.predict_link(X, offset=offsets))
    assert model.score(X, target, offset=offsets, sample_weight=weights) == score_before
