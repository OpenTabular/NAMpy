"""Fisher collection follows the selected model and retains no execution state."""

from dataclasses import fields, replace

import lightning as pl
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from nampy import (
    GPNAMRegressor,
    IGANNClassifier,
    LinRegClassifier,
    LinRegLSS,
    LinRegRegressor,
)
from nampy.diagnostics.fisher import FisherDiagnostics
from nampy.neural.configs.gpnam_config import DefaultGPNAMConfig
from nampy.neural.configs.igann_config import DefaultIGANNConfig
from nampy.neural.configuration import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    IGANNTrainingConfig,
)


def data():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 32), "z": np.cos(np.arange(32))})
    return X, np.resize([0, 1], 32)


def training(**kwargs):
    return GradientTrainingConfig(
        max_epochs=2,
        random_state=7,
        batch_size=8,
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        **kwargs,
    )


def test_constructor_configuration_clones_and_nested_updates():
    config = FisherDiagnostics(n_inputs=3, regression_scale=1.0)
    estimator = LinRegRegressor(fisher_diagnostics=config)
    copied = clone(estimator)
    assert copied.fisher_diagnostics == config
    copied.set_params(fisher_diagnostics__n_draws=4)
    assert copied.fisher_diagnostics.n_draws == 4
    assert estimator.fisher_diagnostics.n_draws == 8
    assert (
        LinRegRegressor()
        .set_params(fisher_diagnostics__n_draws=6)
        .fisher_diagnostics.n_draws
        == 6
    )


@pytest.mark.parametrize("name", ["linreg", "nam", "qnam", "spline_nam", "treenam"])
def test_fisher_term_groups_follow_explicit_architecture_ownership(name):
    from nampy.neural.fisher import ParameterGroups
    from nampy.neural.fitted import NeuralPredictionModule
    from nampy.neural.objectives import RegressionObjective
    from nampy.neural.registry import get_architecture

    spec = get_architecture(name)
    available = {field.name for field in fields(spec.config)}
    options = {"interaction_degree": 2, "layer_sizes": [3], "tree_depth": 2}
    architecture = spec.module(
        config=spec.config(
            **{key: value for key, value in options.items() if key in available}
        ),
        num_feature_info={"x": {"dimension": 1}, "z": {"dimension": 1}},
        cat_feature_info={},
        num_classes=1,
    )
    wrapper = NeuralPredictionModule(architecture, RegressionObjective())
    groups = ParameterGroups.from_module(wrapper, "term")
    expected = {"x", "z", "__global__"}
    if "interaction_degree" in available:
        expected.add("x:z")
    assert groups.grouping == "term"
    assert set(groups.names) == expected
    combined = np.concatenate(groups.indices)
    assert_array_equal(
        np.sort(combined), np.arange(sum(p.numel() for p in wrapper.parameters()))
    )


def test_disabled_diagnostics_perform_no_fisher_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled Fisher diagnostics performed numerical work.")

    monkeypatch.setattr(
        "nampy.neural.training.fisher.estimate_fisher_information", forbidden
    )
    X, y = data()
    estimator = LinRegClassifier(training=training()).fit(X, y)
    assert estimator.fisher_history_ == ()
    assert estimator.fisher_information_ is None
    assert estimator.fisher_config_ is None
    assert not hasattr(estimator, "data_module")


@pytest.mark.parametrize("average", [False, True])
def test_periodic_final_selected_weights_offsets_and_artifact(tmp_path, average):
    callback_events = []

    class Seen(pl.Callback):
        def __init__(self):
            self.epochs = []

        def on_train_epoch_end(self, trainer, pl_module):
            self.epochs.append(int(trainer.current_epoch) + 1)
            callback_events.append(int(trainer.current_epoch) + 1)

    callback = Seen()
    controls = training(average_checkpoints=average, n_last_checkpoints=2)
    controls = replace(
        controls, trainer_kwargs={**controls.trainer_kwargs, "callbacks": [callback]}
    )
    X, y = data()
    V, target = X.iloc[:7].copy(), y[:7]
    offset = np.linspace(-0.2, 0.3, len(X))
    validation_offset = np.linspace(-0.1, 0.1, len(V))
    config = FisherDiagnostics(every_n_epochs=1, n_inputs=4, random_state=9)
    estimator = LinRegClassifier(training=controls, fisher_diagnostics=config).fit(
        X, y, X_val=V, y_val=target, offset=offset, offset_val=validation_offset
    )
    assert callback.epochs == []  # Training owns a deep copy of caller callbacks.
    assert estimator.training_config_.trainer_kwargs["callbacks"][0].epochs == []
    assert callback_events == [1, 2]
    assert [(s.stage, s.epoch) for s in estimator.fisher_history_] == [
        ("initial", None),
        ("epoch", 1),
        ("final", 2),
    ]
    assert estimator.fisher_history_[0].diagonal is None
    assert estimator.fisher_history_[-1].diagonal is not None
    selected = list(estimator.fisher_probe_indices_)
    direct = estimator.fisher_information(
        V.iloc[selected], offset=validation_offset[selected]
    )
    assert_allclose(direct.diagonal, estimator.fisher_information_.diagonal, atol=1e-12)
    assert direct.normalization == "mean"
    assert estimator.best_model_path_ is None
    assert not hasattr(estimator, "data_module")
    assert not hasattr(estimator, "preprocessor")
    restored = LinRegClassifier.load_model(
        estimator.save_model(tmp_path / "fisher.nampy")
    )
    assert_array_equal(restored.fisher_information_.diagonal, direct.diagonal)
    assert_allclose(
        restored.fisher_information(
            V.iloc[selected], offset=validation_offset[selected]
        ).diagonal,
        direct.diagonal,
    )
    with pytest.raises(ValueError, match="offset"):
        restored.fisher_information(V)
    with pytest.raises(ValueError, match="offset"):
        restored.fisher_information(V, offset=np.zeros((len(V), 2)))


def test_offsets_follow_reference_row_permutations_and_change_information():
    X, y = data()
    offset = np.linspace(-1, 1, len(X))
    estimator = LinRegClassifier(training=training()).fit(X, y, offset=offset)
    direct = estimator.fisher_information(X, offset=offset)
    permutation = np.arange(len(X))[::-1]
    permuted = estimator.fisher_information(
        X.iloc[permutation], offset=offset[permutation], batch_size=3
    )
    assert_allclose(direct.diagonal, permuted.diagonal, rtol=1e-7, atol=1e-9)
    saturated = estimator.fisher_information(X, offset=np.full(len(X), 30.0))
    assert np.sum(saturated.diagonal) < np.sum(direct.diagonal) * 0.01


def test_diagnostics_do_not_change_fitted_parameters():
    X, y = data()
    plain = LinRegClassifier(training=training()).fit(X, y)
    checked = LinRegClassifier(
        training=training(),
        fisher_diagnostics=FisherDiagnostics(
            n_inputs=3, estimator="score_outer_product", n_draws=3, random_state=91
        ),
    ).fit(X, y)
    for key, value in plain.model.state_dict().items():
        assert_array_equal(value.cpu(), checked.model.state_dict()[key].cpu())


def test_failed_final_diagnostics_preserve_published_fit(monkeypatch):
    X, y = data()
    estimator = LinRegClassifier(training=training()).fit(X, y)
    before = estimator.predict(X)
    published_model = estimator.model
    estimator.fisher_diagnostics = FisherDiagnostics(include_initial=False)

    def fail(*args, **kwargs):
        raise FloatingPointError("Fisher failed")

    monkeypatch.setattr(
        "nampy.neural.training.fisher.estimate_fisher_information", fail
    )
    with pytest.raises(FloatingPointError, match="Fisher failed"):
        estimator.fit(X, 1 - y)
    assert estimator.model is published_model
    assert_array_equal(estimator.predict(X), before)
    assert not hasattr(estimator, "data_module")


@pytest.mark.parametrize("route", ["native", "fixed"])
def test_native_and_fixed_routes_final_collection_and_periodic_rejection(route):
    X, y = data()
    config = FisherDiagnostics(n_inputs=3, regression_scale=1.0)
    if route == "native":
        estimator = IGANNClassifier(
            architecture_config=DefaultIGANNConfig(n_estimators=1, n_hid=2),
            training=IGANNTrainingConfig(),
            fisher_diagnostics=config,
        )
    else:
        estimator = GPNAMRegressor(
            architecture_config=DefaultGPNAMConfig(rff_num_feat=2),
            training=FixedDesignTrainingConfig(),
            fisher_diagnostics=config,
        )
    estimator.fit(X, y)
    assert estimator.fisher_information_ is not None
    assert estimator.fisher_information(X.iloc[:3]).diagonal is not None
    with pytest.raises(ValueError, match="Periodic"):
        clone(estimator).set_params(
            fisher_diagnostics=replace(config, every_n_epochs=1)
        ).fit(X, y)


def test_distributional_postfit_offsets_use_raw_predictor_width():
    X, _ = data()
    target = np.sin(X.x.to_numpy())
    estimator = LinRegLSS(training=training()).fit(
        X, target, offset=np.zeros((len(X), 2))
    )
    with pytest.raises(ValueError, match="offset"):
        estimator.fisher_information(X)
    with pytest.raises(ValueError, match="offset"):
        estimator.fisher_information(X, offset=np.zeros(len(X)))
    assert (
        estimator.fisher_information(X, offset=np.zeros((len(X), 2))).diagonal
        is not None
    )


def test_negative_hessian_collection_and_population_error_match_exact_information():
    X, _ = data()
    V = X.iloc[:8]
    target = np.sin(X.x.to_numpy())
    config = FisherDiagnostics(
        estimator="negative_hessian",
        input_sampling="iid",
        regression_scale=1.0,
        n_inputs=len(V),
        random_state=11,
        n_draws=3,
    )
    estimator = LinRegRegressor(training=training(), fisher_diagnostics=config).fit(
        X, target, X_val=V, y_val=target[: len(V)]
    )
    sampled = estimator.fisher_information_
    exact = estimator.fisher_information(V, config=replace(config, estimator="exact"))
    assert_allclose(sampled.diagonal, exact.diagonal, rtol=2e-6, atol=2e-7)
    assert sampled.input_estimator_variance is not None
    assert sampled.total_standard_error is not None
    assert np.max(np.abs(sampled.output_sampling_variance)) < 1e-12
    census = estimator.fisher_information(
        V,
        config=replace(config, estimator="exact", input_sampling="without_replacement"),
    )
    assert_allclose(census.input_estimator_variance, 0.0)
    subset = estimator.fisher_information(
        V,
        config=replace(
            config,
            estimator="exact",
            input_sampling="without_replacement",
            population_size=len(X),
        ),
    )
    assert np.any(subset.input_estimator_variance > 0)
