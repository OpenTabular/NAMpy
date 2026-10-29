"""Subject-batched neural interval hazards and existing training integration."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from nampy import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    SurvivalObjectiveConfig,
)
from nampy.models import LinRegSurvival, NAMSurvival
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.registry import get_architecture
from nampy.neural.survival import PiecewiseExponentialObjective
from nampy.survival import PiecewiseExponentialDesign, SurvivalTarget


def _data():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 30)})
    y = SurvivalTarget(
        np.r_[np.linspace(0.1, 0.9, 15), np.linspace(1.1, 2.9, 15)],
        np.tile([1, 0, 1], 10),
        entry=np.full(30, 0.02),
    )
    return X, y


def _training(tmp_path, **changes):
    return GradientTrainingConfig(
        max_epochs=2,
        batch_size=8,
        checkpoint_path=str(tmp_path),
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        **changes,
    )


def test_subject_likelihood_gradients_and_weight_normalization():
    _, y = _data()
    design = PiecewiseExponentialDesign(y, [0, 1, 3])
    packed = np.column_stack([design.exposure, design.events])
    weights = np.tile([0.0, 1.0, 3.0], 10)
    objective = PiecewiseExponentialObjective(2)
    objective.configure_data(
        SimpleNamespace(
            y_train=packed,
            y_val=packed,
            sample_weight_train=weights,
            sample_weight_val=weights,
        )
    )
    eta = torch.full((len(y), 2), -0.3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor(packed)
    w = torch.tensor(weights)
    loss = objective.compute_stage_loss(eta, targets, w, stage="train")
    expected = -weights @ design.log_likelihood(eta.detach().numpy()) / weights.sum()
    assert_allclose(loss.detach(), expected)
    (gradient,) = torch.autograd.grad(loss, eta)
    expected_gradient = (
        weights[:, None]
        * (design.exposure * np.exp(-0.3) - design.events)
        / weights.sum()
    )
    assert_allclose(gradient, expected_gradient, atol=1e-15)
    chunks = [np.arange(0, 7), np.arange(7, 22), np.arange(22, 30)]
    for stage in ("train", "val"):
        total = sum(
            len(rows)
            * objective.compute_stage_loss(
                eta[rows], targets[rows], w[rows], stage=stage
            )
            for rows in chunks
        ) / len(y)
        torch.testing.assert_close(total, loss)
    # Refining a constant-hazard interval must not introduce a row-count divisor.
    fine = PiecewiseExponentialDesign(y, [0, 0.5, 1, 3])
    assert_allclose(
        design.log_likelihood(np.full((len(y), 2), -0.3)),
        fine.log_likelihood(np.full((len(y), 3), -0.3)),
    )


@pytest.mark.parametrize("cls", [LinRegSurvival, NAMSurvival])
def test_gradient_fit_curves_components_and_persistence(cls, tmp_path):
    X, y = _data()
    options = (
        {}
        if cls is LinRegSurvival
        else {"architecture_config": DefaultNAMConfig(layer_sizes=[4], dropout=0)}
    )
    estimator = cls(
        objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)),
        training=_training(tmp_path),
        **options,
    )
    estimator.fit(X, y)
    curves = estimator.predict(X, times=[0, 0.5, 1, 2, 3])
    assert curves.shape == (len(X), 5)
    assert_allclose(curves[:, 0], 1)
    assert (np.diff(curves, axis=1) <= 0).all()
    components = estimator.predict_components(X)
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(X), 2)
    assert_allclose(
        np.exp(components.link),
        estimator.predict_distribution(X).hazard_rates,
        rtol=1e-6,
    )
    assert np.isfinite(estimator.score(X, y))
    assert estimator.training_rows_ + estimator.validation_rows_ == len(y)
    assert not hasattr(estimator, "data_module")
    restored = cls.load_model(
        estimator.save_model(tmp_path / (cls.__name__ + ".nampy"))
    )
    assert_array_equal(curves, restored.predict(X, times=[0, 0.5, 1, 2, 3]))
    assert_array_equal(components.link, restored.predict_components(X).link)


def test_configuration_routes_and_transactional_failures(tmp_path, monkeypatch):
    X, y = _data()
    objective = SurvivalObjectiveConfig(cuts=(0, 1, 3))
    training = _training(tmp_path)
    estimator = LinRegSurvival(objective=objective, training=training).fit(X, y)
    before = estimator.predict(X, times=[1, 3])
    module = estimator.model

    def fail(*args, **kwargs):
        raise RuntimeError("injected survival failure")

    monkeypatch.setattr("nampy.neural.training.engine.fit_gradient", fail)
    with pytest.raises(RuntimeError, match="injected"):
        estimator.fit(X, y)
    assert estimator.model is module
    assert_array_equal(before, estimator.predict(X, times=[1, 3]))
    with pytest.raises(NotImplementedError, match="gradient"):
        LinRegSurvival(objective=objective, training=FixedDesignTrainingConfig()).fit(
            X, y
        )
    with pytest.raises(NotImplementedError, match="custom samplers"):
        LinRegSurvival(
            objective=objective,
            training=_training(tmp_path, dataloader_kwargs={"drop_last": True}),
        ).fit(X, y)
    with pytest.raises(TypeError, match="does not support"):
        get_architecture("igann").require("survival")
    new = clone(estimator).set_params(training__lr=0.02)
    assert new.model is None and new.training.lr == 0.02
    assert training.lr != 0.02
    with pytest.raises(NotImplementedError, match="Supported"):
        SurvivalObjectiveConfig(likelihood="efron")


def test_unsupervised_preprocessing_uses_training_subjects_only(tmp_path):
    from nampy import PreprocessingConfig

    X, y = _data()
    X_val = pd.DataFrame({"x": [200.0, 300.0]})
    y_val = SurvivalTarget([0.5, 2.0], [0, 1])
    estimator = LinRegSurvival(
        objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)),
        preprocessing=PreprocessingConfig(numerical_preprocessing="standardization"),
        training=_training(tmp_path),
    ).fit(X, y, X_val=X_val, y_val=y_val)
    transformed = estimator.fitted_preprocessing_.preprocess_tensors(X)[1]["x"].numpy()
    assert_allclose(transformed.mean(), 0, atol=1e-7)
    assert_allclose(transformed.std(), 1, atol=1e-7)
    assert estimator.training_rows_ == len(y)
    assert estimator.validation_rows_ == 2


def test_neural_constant_hazards_reach_weighted_survival_mle(tmp_path):
    X, y = _data()
    X["x"] = 0.0
    weights = np.tile([1.0, 2.0, 3.0], 10)
    training = GradientTrainingConfig(
        max_epochs=300,
        batch_size=len(y),
        lr=0.05,
        lr_schedule="none",
        patience=300,
        average_checkpoints=True,
        weight_decay=0,
        checkpoint_path=str(tmp_path),
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    estimator = LinRegSurvival(
        objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)), training=training
    )
    estimator.fit(
        X, y, X_val=X.copy(), y_val=y, sample_weight=weights, sample_weight_val=weights
    )
    design = PiecewiseExponentialDesign(y, [0, 1, 3])
    mle = (weights @ design.events) / (weights @ design.exposure)
    assert_allclose(estimator.predict_distribution(X).hazard_rates[0], mle, rtol=2e-4)


def test_explicit_subject_ids_and_prediction_offsets(tmp_path):
    X, y = _data()
    objective = SurvivalObjectiveConfig(cuts=(0, 1, 3))
    offset = np.linspace(-0.1, 0.1, len(y))
    estimator = LinRegSurvival(objective=objective, training=_training(tmp_path)).fit(
        X, y, offset=offset
    )
    with pytest.raises(ValueError, match="offset"):
        estimator.predict(X, times=[1])
    components = estimator.predict_components(X, offset=offset)
    components.validate_additive_reconstruction()
    assert_allclose(components.link, estimator.predict_link(X, offset=offset))
    centered = estimator.predict_components(
        X, offset=offset, center=True, reference_X=X
    )
    centered.validate_additive_reconstruction()
    for contribution in centered.terms.values():
        assert_allclose(contribution.mean(axis=0), 0, atol=1e-7)
    missing = X.copy()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        estimator.predict(missing, times=[1], offset=offset)
    explicit = SurvivalTarget(y.time, y.event, subject_id=np.arange(len(y)))
    with pytest.raises(ValueError, match="IDs overlap"):
        LinRegSurvival(objective=objective).fit(X, explicit, X_val=X, y_val=explicit)
