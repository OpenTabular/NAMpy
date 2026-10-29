"""Fitted prediction state, transactional refits and route-specific ownership."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from nampy import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    IGANNTrainingConfig,
    NativeTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)
from nampy.models.gpnam import GPNAMRegressor
from nampy.models.igann import IGANNLSS, IGANNClassifier
from nampy.models.linreg import LinRegLSS, LinRegRegressor
from nampy.neural.configs.gpnam_config import DefaultGPNAMConfig
from nampy.neural.fitted import NeuralPredictionModule


def _data():
    x = np.linspace(-1, 1, 24)
    return pd.DataFrame({"x": x}), 0.5 + x**2


def _gradient(tmp_path, **changes):
    return GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path),
                                  trainer_kwargs={"logger": False, "enable_progress_bar": False,
                                                  "enable_model_summary": False}, **changes)


@pytest.mark.parametrize("cls", [LinRegRegressor, LinRegLSS])
def test_failed_refit_preserves_all_published_prediction_state(cls, tmp_path, monkeypatch):
    X, y = _data()
    estimator = cls(training=_gradient(tmp_path)).fit(X, y)
    before = estimator.predict(X).copy()
    fitted = estimator.model
    schema = estimator.schema_

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", fail)
    with pytest.raises(RuntimeError, match="injected"):
        estimator.fit(X, -y)
    assert estimator.model is fitted
    assert estimator.schema_ is schema
    np.testing.assert_array_equal(estimator.predict(X), before)
    if cls is LinRegLSS:
        assert estimator.distribution_fit_status().eligible


def test_fitted_artifact_has_prediction_state_without_training_runtime(tmp_path):
    X, y = _data()
    estimator = LinRegRegressor(training=_gradient(tmp_path)).fit(X, y)
    assert isinstance(estimator.model, NeuralPredictionModule)
    assert not hasattr(estimator, "data_module")
    assert not hasattr(estimator.model, "_trainer")
    assert not hasattr(estimator.fitted_preprocessing_, "X_train")
    restored = LinRegRegressor.load_model(estimator.save_model(tmp_path / "fit.nampy"))
    np.testing.assert_array_equal(estimator.predict(X), restored.predict(X))
    np.testing.assert_array_equal(estimator.predict_components(X).link,
                                  restored.predict_components(X).link)


def test_nested_configuration_updates_do_not_mutate_caller_and_invalidate_fit(tmp_path):
    X, y = _data()
    training = _gradient(tmp_path)
    preprocessing = PreprocessingConfig(numerical_preprocessing="standardization")
    estimator = LinRegRegressor(training=training, preprocessing=preprocessing).fit(X, y)
    estimator.set_params(training__lr=0.02, preprocessing__n_bins=9)
    assert training.lr != 0.02
    assert "n_bins" not in preprocessing.parameters
    assert estimator.model is None
    assert clone(estimator).training.lr == 0.02


def test_empty_parameter_update_preserves_a_successful_fit(tmp_path):
    X, y = _data()
    estimator = LinRegRegressor(training=_gradient(tmp_path)).fit(X, y)
    module = estimator.model
    assert estimator.set_params() is estimator
    assert estimator.model is module


def test_nested_training_updates_select_the_default_fixed_or_native_record():
    from nampy.models import IGANNRegressor

    fixed = GPNAMRegressor().set_params(training__ridge=0.25)
    assert isinstance(fixed.training, FixedDesignTrainingConfig)
    assert fixed.training.ridge == 0.25
    native = IGANNRegressor().set_params(training__boost_rate=0.3)
    assert isinstance(native.training, IGANNTrainingConfig)
    assert native.training.boost_rate == 0.3
    gradient = GPNAMRegressor().set_params(training__lr=0.005)
    assert isinstance(gradient.training, GradientTrainingConfig)


def test_artifact_excludes_execution_objects_without_mutating_configuration(tmp_path):
    X, y = _data()
    estimator = LinRegRegressor(training=_gradient(tmp_path)).fit(X, y)
    def callback():
        return None
    options = {**estimator.training.trainer_kwargs, "callbacks": [callback]}
    estimator.training = replace(estimator.training, trainer_kwargs=options)
    restored = LinRegRegressor.load_model(estimator.save_model(tmp_path / "callbacks.nampy"))
    assert estimator.training.trainer_kwargs["callbacks"] == [callback]
    assert "callbacks" not in restored.training.trainer_kwargs
    np.testing.assert_array_equal(estimator.predict(X), restored.predict(X))


def test_route_resolution_uses_architecture_objective_and_actual_class_count():
    X, _ = _data()
    binary = IGANNClassifier()._resolved_training(np.tile([0, 1], 12))
    multiclass = IGANNClassifier()._resolved_training(np.tile([0, 1, 2], 8))
    assert isinstance(binary, IGANNTrainingConfig)
    assert isinstance(multiclass, GradientTrainingConfig)
    assert not hasattr(binary, "lr")
    with pytest.raises(NotImplementedError, match="Native training"):
        IGANNLSS(training=IGANNTrainingConfig()).fit(X, X.x)
    with pytest.raises(NotImplementedError, match="Fixed-design"):
        LinRegRegressor(training=FixedDesignTrainingConfig()).fit(X, X.x)
    custom = GPNAMRegressor(objective=ObjectiveConfig(loss_fct=lambda p, y: (p-y)**2))
    assert isinstance(custom._resolved_training(X.x), GradientTrainingConfig)


def test_native_route_requires_the_architectures_training_record():
    with pytest.raises(TypeError, match="requires IGANNTrainingConfig"):
        IGANNClassifier(training=NativeTrainingConfig())._resolved_training([0, 1])


def test_routes_reject_resampling_controls_that_would_be_ignored():
    with pytest.raises(ValueError, match="sampling_strategy"):
        ObjectiveConfig(sampling_strategy="unimplemented")
    with pytest.raises(NotImplementedError, match="Resampling requires"):
        IGANNClassifier(
            objective=ObjectiveConfig(sampling_strategy="balanced")
        )._resolved_training([0, 1])
    with pytest.raises(NotImplementedError, match="Resampling requires"):
        GPNAMRegressor(training=FixedDesignTrainingConfig(
            dataloader_kwargs={"sampler": object()}
        ))._resolved_training([0.0, 1.0])


def test_fixed_design_preserves_all_training_rows_and_preprocessing(tmp_path):
    X, y = _data()
    estimator = GPNAMRegressor(architecture_config=DefaultGPNAMConfig(rff_num_feat=5),
                               training=FixedDesignTrainingConfig()).fit(X, y)
    assert estimator.training_rows_ == len(X)
    assert estimator.linear_solver_info_["n_rows"] == len(X)
    restored = GPNAMRegressor.load_model(estimator.save_model(tmp_path / "fixed.nampy"))
    np.testing.assert_array_equal(estimator.predict(X), restored.predict(X))


def test_failed_warm_start_changes_only_staged_weights(tmp_path, monkeypatch):
    X, y = _data()
    estimator = LinRegRegressor(training=_gradient(tmp_path, warm_start=True)).fit(X, y)
    before = estimator.predict(X).copy()

    def corrupt_then_fail(staged, training, *, fisher=None):
        for parameter in staged.model.parameters():
            parameter.data.fill_(999)
        raise RuntimeError("warm fit failed")

    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", corrupt_then_fail)
    with pytest.raises(RuntimeError, match="warm fit failed"):
        estimator.fit(X, y)
    np.testing.assert_array_equal(estimator.predict(X), before)


def test_runtime_loss_is_owned_by_the_fit_and_failed_refits_cannot_mutate_it(tmp_path, monkeypatch):
    import torch

    class TrainableLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.zeros(()))

        def forward(self, prediction, target):
            return (prediction - target - self.shift) ** 2

    X, y = _data()
    caller_loss = TrainableLoss()
    estimator = LinRegRegressor(objective=ObjectiveConfig(loss_fct=caller_loss),
                               training=_gradient(tmp_path)).fit(X, y)
    assert caller_loss.shift.item() == 0
    runtime_loss = estimator.model.objective.loss_fct
    assert runtime_loss is not caller_loss
    previous = runtime_loss.shift.detach().clone()

    def fail(staged, training, *, fisher=None):
        staged.model.objective.loss_fct.shift.data.fill_(999)
        raise RuntimeError("loss mutation during failed fit")

    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", fail)
    with pytest.raises(RuntimeError, match="loss mutation"):
        estimator.fit(X, y)
    torch.testing.assert_close(runtime_loss.shift, previous)
    assert caller_loss.shift.item() == 0


@pytest.mark.parametrize("cls", [LinRegRegressor, IGANNClassifier])
def test_distribution_settings_cannot_be_silently_ignored_by_point_estimators(cls):
    with pytest.raises(ValueError, match="require an LSS estimator"):
        cls(objective=ObjectiveConfig(family="gamma"))
