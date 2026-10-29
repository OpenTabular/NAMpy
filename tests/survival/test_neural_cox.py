"""Full risk-set neural Cox fitting and mgcv Breslow numerical contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from nampy import GradientTrainingConfig, SurvivalObjectiveConfig
from nampy.gam.losses.cox import cox_partial_likelihood
from nampy.models import LinRegSurvival, NAMSurvival
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.survival import CoxObjective
from nampy.survival import SurvivalTarget
from nampy.survival.cox import baseline_from_predictors


def test_cox_objective_derivatives_match_mgcv_risk_set_owner():
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(12, 3))
    time = np.resize([1., 1., 2., 3.], 12)
    event = np.resize([1, 0, 1, 1], 12).astype(bool)
    offset = rng.normal(size=12) * .05
    beta = np.array([.2, -.1, .3])
    targets = torch.tensor(np.column_stack([time, event, np.zeros(12)]))
    objective = CoxObjective()

    def value(b):
        eta = torch.tensor(matrix) @ b + torch.tensor(offset)
        return -12 * objective.compute_loss(eta[:, None], targets)

    expected, gradient, hessian = cox_partial_likelihood(
        beta, matrix, time, event, np.zeros(12), offset
    )
    b = torch.tensor(beta, requires_grad=True)
    assert_allclose(value(b).detach(), expected, atol=1e-13)
    assert_allclose(torch.func.grad(value)(b).detach(), gradient, atol=1e-13)
    assert_allclose(torch.func.hessian(value)(b).detach(), hessian, atol=1e-13)
    # The Cox intercept is unidentifiable; logsumexp remains stable at large shifts.
    eta = torch.tensor(matrix @ beta + offset)[:, None]
    assert_allclose(objective.compute_loss(eta + 10000, targets), -expected / 12, atol=2e-12)


def test_neural_cox_matches_static_mgcv_likelihood_gradient_and_baseline():
    fixture = pd.read_csv(Path(__file__).parents[1] / "reference_fixtures/neural_cox.csv")
    objective = CoxObjective()
    eta = torch.tensor(fixture.eta.to_numpy(), requires_grad=True)
    targets = torch.tensor(np.column_stack([fixture.time, fixture.event, np.zeros(len(fixture))]))
    value = -len(fixture) * objective.compute_loss(eta[:, None], targets)
    assert_allclose(value.detach(), fixture.log_likelihood.iloc[0], atol=2e-14)
    assert_allclose(torch.autograd.grad(value, eta)[0], fixture.gradient, atol=2e-14)
    times, baseline = baseline_from_predictors(fixture.eta.to_numpy(), fixture.time.to_numpy(),
        fixture.event.to_numpy(), np.zeros(len(fixture)))[0]
    assert_allclose(baseline[np.searchsorted(times, fixture.time)], fixture.baseline, atol=2e-14)


def test_configure_data_forces_complete_cohorts_and_rejects_partial_loss():
    packed = np.array([[1., 1., 0.], [1., 0., 0.], [2., 1., 0.]])
    data = SimpleNamespace(y_train=packed, y_val=packed[:2],
                           sample_weight_train=None, sample_weight_val=None, batch_size=1)
    objective = CoxObjective()
    objective.configure_data(data)
    assert data.batch_size == 3
    with pytest.raises(ValueError, match="complete"):
        objective.compute_stage_loss(torch.zeros(2, 1), torch.tensor(packed[:2]), stage="train")
    for rows, weights, error, match in (
        (packed + [0., 0., .1], np.ones(3), NotImplementedError, "entry"),
        (packed, np.array([1., 2., 1.]), NotImplementedError, "weights"),
        (packed * [1., 0., 1.], np.ones(3), ValueError, "event"),
    ):
        with pytest.raises(error, match=match):
            objective.validate_training_targets(rows, weights)


@pytest.mark.parametrize("cls", [LinRegSurvival, NAMSurvival])
def test_neural_cox_fit_baseline_offsets_components_and_persistence(cls, tmp_path):
    source = pd.read_csv(Path(__file__).parents[1] / "reference_fixtures/cox_fixed_input.csv")
    X = source[["x", "z"]]
    y = SurvivalTarget(source.time, source.event)
    options = {} if cls is LinRegSurvival else {
        "architecture_config": DefaultNAMConfig(layer_sizes=[4], dropout=0)
    }
    training = GradientTrainingConfig(max_epochs=2, batch_size=2,
        trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})
    offset = source.offset.to_numpy()
    # Explicit validation has different follow-up: it must not enter the baseline.
    val = SurvivalTarget(source.time + 20, source.event)
    estimator = cls(objective=SurvivalObjectiveConfig(likelihood="cox"), training=training,
                    **options).fit(X, y, X_val=X, y_val=val, offset=offset, offset_val=offset)
    assert estimator.n_optimizer_steps_ == 2
    assert estimator.training_rows_ == len(y)
    assert estimator.best_model_path_ is None
    assert estimator.baseline_times_[-1] == y.time.max()
    eta = estimator.predict_link(X, offset=offset)[:, 0]
    times, cumulative = baseline_from_predictors(eta - eta.max(), y.time, y.event, np.zeros(len(y)))[0]
    assert_allclose(estimator.baseline_times_, times)
    assert_allclose(estimator.baseline_cumulative_hazard_, cumulative)
    prediction = estimator.predict_distribution(X, offset=offset)
    expected = np.exp(eta - eta.max())[:, None] * cumulative
    assert_allclose(prediction.cumulative_hazard_steps, expected)
    assert (np.diff(prediction.survival(times), axis=1) <= 0).all()
    # Float32 architecture addition is evaluated before float64 known offsets.
    estimator.predict_components(X, offset=offset).validate_additive_reconstruction(
        rtol=16 * np.finfo(np.float32).eps, atol=16 * np.finfo(np.float32).eps
    )
    assert np.isfinite(estimator.score(X, y, offset=offset))
    with pytest.raises(ValueError, match="offset"):
        estimator.predict_link(X)
    restored = cls.load_model(estimator.save_model(tmp_path / f"{cls.__name__}.nampy"))
    assert_array_equal(prediction.survival(times), restored.predict(X, times=times, offset=offset))
    assert not hasattr(estimator, "data_module")


def test_cox_rejects_partial_trainer_limits():
    estimator = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood="cox"),
        training=GradientTrainingConfig(trainer_kwargs={"limit_train_batches": .5}))
    X = pd.DataFrame({"x": [0., 1., 2.]})
    with pytest.raises(NotImplementedError, match="complete"):
        estimator.fit(X, SurvivalTarget([1, 2, 3], [1, 0, 1]))


@pytest.mark.parametrize("predictors", [[-1000., 0.], [np.nan, 0.], [710., 0.], [709.7, 709.7]])
def test_cox_baseline_rejects_nonrepresentable_risk_weights_and_totals(predictors):
    with pytest.raises(FloatingPointError):
        baseline_from_predictors(np.array(predictors), np.array([2., 1.]),
                                 np.array([1, 1]), np.zeros(2))


def test_neural_cox_rejects_custom_ties_and_strata():
    with pytest.raises(TypeError, match="ties"):
        SurvivalObjectiveConfig(likelihood="cox", ties="efron")
    estimator = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood="cox"))
    with pytest.raises(TypeError, match="strata"):
        estimator.fit(pd.DataFrame({"x": [0., 1.]}),
                      SurvivalTarget([1., 2.], [1, 1]), strata=[0, 1])
