"""Task contracts are explicit while backend parameterizations stay independent."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy import GAMLSS, GradientTrainingConfig, LinRegLSS, ObjectiveConfig
from nampy.contracts import CapabilityStatus
from nampy.diagnostics import FisherDiagnostics, prepare_survival_effects
from nampy.models import GAMRegressor


@pytest.fixture(params=["statistical", "neural"])
def fitted_normal(request):
    X = pd.DataFrame({"x": np.linspace(-1, 1, 32)})
    y = 0.2 + X.x.to_numpy() + 0.3 * np.cos(np.arange(len(X)))
    if request.param == "statistical":
        model = GAMLSS(formula={"mu": "~x", "sigma": "~1"}, optimize_smoothing=False)
    else:
        model = LinRegLSS(
            training=GradientTrainingConfig(
                max_epochs=1,
                trainer_kwargs={
                    "logger": False,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
            )
        )
    return model.fit(X, y), X, y


def test_probability_parameters_and_task_prediction_have_distinct_meanings(
    fitted_normal,
):
    model, X, y = fitted_normal
    parameters = model.predict_parameters(X)
    assert parameters.shape == (len(X), len(model.parameter_names_))
    assert model.predict_link(X).shape == (len(X), len(model.predictor_names_))
    assert_allclose(model.predict(X), model.predict_distribution(X).mean(), rtol=1e-6)
    assert_allclose(model.predict(X), model.predict_point(X), rtol=1e-6)
    assert_allclose(
        model.score(X, y), model.predict_distribution(X).log_prob(y).mean(), rtol=1e-5
    )
    with pytest.raises(TypeError):
        model.predict(X, raw=True)
    with pytest.raises(ValueError):
        model.predict(None)


def test_offsets_share_row_predictor_axes_and_batch_semantics(fitted_normal):
    model, X, _ = fitted_normal
    offset = np.column_stack([np.linspace(-0.1, 0.1, len(X)), np.full(len(X), 0.02)])
    raw = model.predict_link(X)
    shifted = model.predict_link(X, offset=offset, batch_size=5)
    assert_allclose(shifted, raw + offset, atol=2e-7, rtol=1e-6)
    permutation = np.random.default_rng(7).permutation(len(X))
    assert_allclose(
        model.predict_link(
            X.iloc[permutation], offset=offset[permutation], batch_size=3
        ),
        shifted[permutation],
        atol=2e-7,
        rtol=1e-6,
    )
    for wrong in (offset.T, offset[:, 0], np.zeros((len(X), 1))):
        with pytest.raises(ValueError, match="shape"):
            model.predict_link(X, offset=wrong)


def test_fisher_uses_one_public_configuration_contract(fitted_normal):
    model, X, _ = fitted_normal
    result = model.fisher_information(
        X, config=FisherDiagnostics(record_diagonal="never"), batch_size=5
    )
    assert result.diagonal is None and result.n_inputs == len(X)
    assert result.normalization == "mean"
    with pytest.raises(TypeError):
        model.fisher_information(X, return_diagonal=False)
    with pytest.raises(TypeError):
        model.fisher_information()


def test_capability_discovery_reuses_operation_eligibility(fitted_normal):
    model, X, _ = fitted_normal
    import torch

    before = torch.random.get_rng_state().clone()
    status = model.capabilities()
    assert status["predict_distribution"].state == "supported"
    assert status["fisher.exact"].state == "supported"
    assert isinstance(status["predict_link"], CapabilityStatus)
    with pytest.raises(TypeError):
        status["invented"] = CapabilityStatus("supported")
    assert torch.equal(before, torch.random.get_rng_state())
    assert callable(prepare_survival_effects)
    assert model.predict(X).shape == (len(X),)


def test_quasi_and_quantile_tasks_do_not_acquire_probability_information():
    assert (
        GAMRegressor(family="quasipoisson").capabilities()["fisher.exact"].state
        == "unsupported"
    )
    model = LinRegLSS(
        objective=ObjectiveConfig(
            family="quantile", distributional_kwargs={"quantiles": [0.2, 0.8]}
        )
    )
    status = model.capabilities()
    assert status["predict_parameters"].state == "unsupported"
    assert status["predict_distribution"].state == "unsupported"
    assert status["fisher.score_outer_product"].state == "unsupported"
    assert status["predict_quantiles"].state == "conditional"


@pytest.mark.parametrize('family,kwargs,exact,sampled', [
    ('normal', {}, 'conditional', 'conditional'),
    ('studentt', {}, 'unsupported', 'conditional'),
    ('robustnormal', {}, 'conditional', 'conditional'),
    ('robustnormal', {'rob': 0.5}, 'unsupported', 'unsupported'),
])
def test_unfitted_neural_family_discovery_matches_likelihood_declarations(family, kwargs, exact, sampled):
    model = LinRegLSS(objective=ObjectiveConfig(family=family, distributional_kwargs=kwargs))
    status = model.capabilities()
    assert status['fisher.exact'].state == exact
    assert status['fisher.score_outer_product'].state == sampled
    assert model.model is None


def test_scalar_statistical_predictors_have_explicit_output_axes():
    X = pd.DataFrame({'x': np.linspace(-1, 1, 20)})
    model = GAMRegressor(formula='~x', optimize_smoothing=False).fit(X, np.sin(X.x))
    assert model.predict_link(X).shape == (len(X), 1)
    assert len(model.predictor_names_) == 1
    assert_allclose(model.predict_link(X, offset=np.ones((len(X), 1))),
                    model.predict_link(X) + 1)
    assert model.capabilities()['uncertainty.coefficients'].state == 'supported'


def test_survival_capabilities_preserve_task_and_likelihood_restrictions():
    from nampy import CoxGAM, LinRegSurvival, SurvivalGAM, SurvivalObjectiveConfig

    for model in (CoxGAM(), LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood='cox'))):
        status = model.capabilities()
        assert status['survival.curves'].state == 'conditional'
        assert status['survival.hazard'].state == 'unsupported'
        assert status['fisher.exact'].state == 'unsupported'
    status = SurvivalGAM().capabilities()
    assert status['survival.hazard'].state == 'conditional'
    assert status['survival.intervals'].state == 'conditional'
    assert 'available_covariance' in status['survival.intervals'].conditions
