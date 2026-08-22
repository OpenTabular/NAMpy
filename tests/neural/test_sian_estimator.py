from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from nampy.models import SIANLSS, SIANClassifier, SIANRegressor


def test_sian_estimator_discovers_terms_then_uses_shared_training_engine(tmp_path):
    rng = np.random.default_rng(9)
    X = pd.DataFrame(
        rng.normal(size=(48, 3)), columns=["a", "b", "c"]
    )
    y = X["a"].to_numpy() * X["b"].to_numpy() + 0.1 * X["c"].to_numpy()
    estimator = SIANRegressor(
        layer_sizes=[6],
        reference_layer_sizes=[8],
        reference_epochs=2,
        reference_batch_size=16,
        max_interaction_order=2,
        interaction_thresholds=1.0,
        threshold_mode="fraction",
        selection_max_samples=8,
        selection_max_pairs=12,
        selection_batch_size=64,
        l1_regularization=1e-5,
    )
    estimator.fit(
        X,
        y,
        max_epochs=1,
        batch_size=16,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    assert estimator.interaction_selection_result_ is not None
    assert set(estimator.selected_interactions_) == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }
    assert estimator.fitted_config_.interactions == estimator.selected_interactions_
    assert estimator.config.interactions is None
    assert estimator.model.optimizer_name == "adagrad"
    selection_table = estimator.interaction_selection_table()
    assert set(selection_table["interaction"]) == {"a:b", "a:c", "b:c"}
    assert selection_table["selected"].all()
    components = estimator.predict_components(X.iloc[:6])
    components.validate_additive_reconstruction(rtol=1e-5, atol=1e-7)
    complexity = estimator.model_complexity()
    assert complexity["interaction_terms"] == 3
    assert complexity["active_parameters"] < complexity["total_parameters"]
    expected = estimator.predict(X.iloc[:6])
    assert estimator.compress_terms() is estimator
    np.testing.assert_allclose(estimator.predict(X.iloc[:6]), expected, rtol=1e-6)
    assert estimator.block_mask_terms() is estimator
    np.testing.assert_allclose(estimator.predict(X.iloc[:6]), expected, rtol=1e-6)


def test_sian_explicit_interactions_bypass_reference_selection():
    estimator = SIANRegressor(interactions=(("b", "a"),))
    features = {
        "a": torch.tensor([[0.0], [1.0]]),
        "b": torch.tensor([[1.0], [0.0]]),
    }
    resolved = estimator._prepare_architecture_config(
        estimator.config,
        train_num_features=features,
        train_cat_features={},
        train_targets=torch.tensor([0.0, 1.0]),
        val_num_features=features,
        val_cat_features={},
        val_targets=torch.tensor([0.0, 1.0]),
        objective=type("Objective", (), {"kind": "regression"})(),
    )
    assert resolved is estimator.config
    assert estimator.selected_interactions_ == (("a", "b"),)
    assert estimator.interaction_reference_model_ is None
    assert SIANClassifier().architecture_name == "sian"
    assert SIANLSS().architecture_name == "sian"


def test_sian_lss_is_generated_from_the_same_architecture(tmp_path):
    X = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 32),
            "z": np.linspace(1.0, -1.0, 32),
        }
    )
    y = np.exp(0.2 * X["x"].to_numpy())
    estimator = SIANLSS(
        family="normal",
        interactions=(("x", "z"),),
        layer_sizes=[5],
        l1_regularization=0.0,
    )
    estimator.fit(
        X,
        y,
        max_epochs=1,
        batch_size=16,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    assert estimator.model.model.num_classes == estimator.family_.param_count
    assert estimator.predict(X.iloc[:4]).shape == (4, estimator.family_.param_count)


def test_sian_classifier_uses_discovery_and_shared_binary_objective(tmp_path):
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    y = (X["a"].to_numpy() * X["b"].to_numpy() > 0).astype(int)
    estimator = SIANClassifier(
        layer_sizes=[4],
        reference_layer_sizes=[6],
        reference_epochs=1,
        max_interaction_order=2,
        interaction_thresholds=1.0,
        threshold_mode="fraction",
        selection_max_samples=6,
        selection_max_pairs=8,
        l1_regularization=0.0,
    )
    estimator.fit(
        X,
        y,
        max_epochs=1,
        batch_size=16,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    assert estimator.selected_interactions_ == (("a", "b"),)
    assert estimator.predict_proba(X.iloc[:5]).shape == (5, 2)
