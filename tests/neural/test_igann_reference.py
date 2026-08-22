from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Lasso

from nampy.models.igann import IGANNLSS, IGANNClassifier, IGANNRegressor
from nampy.neural.architectures.igann import IGANN
from nampy.neural.configs.igann_config import DefaultIGANNConfig
from nampy.neural.objectives import BinaryObjective, RegressionObjective


def _numeric_info(*names):
    return {name: {"dimension": 1, "n_unique": 8} for name in names}


def test_igann_random_hidden_blocks_match_upstream_masked_draw():
    config = DefaultIGANNConfig(
        n_hid=3,
        n_estimators=2,
        elm_scale=1.7,
        elm_random_state=0,
    )
    model = IGANN({}, _numeric_info("x", "z"), config=config)

    generator = torch.Generator().manual_seed(0)
    full = torch.randn(2, 6, generator=generator) * 1.7
    expected = torch.stack((full[0, :3], full[1, 3:]))

    torch.testing.assert_close(model.hidden_weights[0], expected)
    assert not model.hidden_weights.requires_grad


def test_igann_reference_codes_categorical_inputs_and_groups_output_terms():
    model = IGANN(
        {"group": {"dimension": 1, "n_unique": 3}},
        {},
        config=DefaultIGANNConfig(n_estimators=0),
    )
    design = model._design_inputs(
        {}, {"group": torch.tensor([[1], [2], [3], [0]])}
    )

    torch.testing.assert_close(
        design,
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
    )
    result = model({}, {"group": torch.tensor([[1], [2], [3], [0]])})
    assert set(result) == {"output", "group", "intercept"}


def test_native_regression_stage_matches_upstream_elm_ridge_equations():
    config = DefaultIGANNConfig(
        n_hid=2,
        n_estimators=1,
        boost_rate=0.2,
        init_reg=0.01,
        elm_alpha=0.5,
        early_stopping=0,
    )
    model = IGANN({}, _numeric_info("x", "z"), config=config)
    train_num = {
        "x": torch.tensor([[-1.0], [-0.3], [0.2], [0.7], [1.1], [1.5]]),
        "z": torch.tensor([[0.5], [-0.5], [1.0], [-1.0], [0.2], [-0.2]]),
    }
    val_num = {
        "x": torch.tensor([[-0.8], [0.9]]),
        "z": torch.tensor([[0.1], [-0.4]]),
    }
    y = torch.tensor([[-1.2], [-0.1], [0.6], [0.4], [1.4], [1.0]])
    y_val = torch.tensor([[-0.7], [1.1]])

    model.fit_native(
        train_num_features=train_num,
        train_cat_features={},
        train_targets=y,
        val_num_features=val_num,
        val_cat_features={},
        val_targets=y_val,
        objective=RegressionObjective(),
        random_state=7,
    )

    design = model._design_inputs(train_num, {})
    baseline = Lasso(alpha=0.01, random_state=7).fit(
        design.numpy(), y[:, 0].numpy()
    )
    initial = torch.as_tensor(
        baseline.predict(design.numpy()), dtype=torch.float32
    ).unsqueeze(-1)
    hidden = model._basis_for_stage(design, 0)
    weighted_hidden = hidden * 0.2
    expected = torch.linalg.solve(
        weighted_hidden.T @ weighted_hidden
        + 0.5 * torch.eye(weighted_hidden.shape[1]),
        weighted_hidden.T @ (y - initial),
    )

    torch.testing.assert_close(model.stage_coefficients[0], expected)
    assert model.n_estimators_ == 1


def test_registered_igann_regression_exposes_components_history_and_complexity():
    generator = np.random.default_rng(4)
    X = pd.DataFrame(generator.normal(size=(48, 3)), columns=["x", "z", "w"])
    y = 1.2 * X["x"] + np.sin(2.0 * X["z"]) - 0.4 * X["w"]
    model = IGANNRegressor(
        n_hid=4,
        n_estimators=3,
        early_stopping=0,
        init_reg=0.01,
        elm_alpha=1.0,
    )

    model.fit(X, y, random_state=11)
    prediction = model.predict_components(X.iloc[:9], batch_size=4)

    reconstructed = (
        np.asarray(prediction.intercept) + sum(prediction.terms.values())
    ).squeeze(-1)
    np.testing.assert_allclose(reconstructed, prediction.link, rtol=1e-5, atol=1e-6)
    assert model.n_estimators_ == 3
    assert len(model.training_history()["val_loss"]) == 3
    assert model.native_training_info_["algorithm"] == "igann_native_elm_boosting"
    assert model.model_complexity()["fitted_estimators"] == 3
    assert model.basis_metadata()["hidden_weights"].shape == (3, 3, 4)


def test_registered_igann_native_binary_and_generic_multiclass_classification():
    X = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.0, 50),
            "z": np.tile(["a", "b"], 25),
        }
    )
    y = np.where(X["x"].to_numpy() + (X["z"] == "b") * 0.3 > 0, "yes", "no")
    model = IGANNClassifier(
        n_hid=3,
        n_estimators=2,
        early_stopping=0,
        init_reg=0.1,
    )
    model.fit(X, y, random_state=3)

    probabilities = model.predict_proba(X.iloc[:7], batch_size=3)
    assert probabilities.shape == (7, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(model.predict(X.iloc[:7])) <= {"no", "yes"}

    multiclass = IGANNClassifier(n_estimators=1, n_hid=2)
    multiclass.fit(
        X.iloc[:12],
        np.tile([0, 1, 2], 4),
        random_state=2,
        max_epochs=1,
        logger=False,
        enable_model_summary=False,
    )
    assert multiclass.native_training_info_ is None
    assert multiclass.predict_proba(X.iloc[:3]).shape == (3, 3)


def test_igann_lss_uses_generic_distribution_objective():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 36)})
    y = 0.5 + X["x"] ** 2
    model = IGANNLSS(
        family="normal",
        n_hid=2,
        n_estimators=2,
    )
    model.fit(
        X,
        y,
        random_state=6,
        max_epochs=2,
        logger=False,
        enable_model_summary=False,
    )

    prediction = model.predict_components(X.iloc[:5])
    assert prediction.link.shape == (5, 2)
    assert prediction.response.shape == (5, 2)
    assert model.native_training_info_ is None
    assert model.n_estimators_ == 2

    native_only = IGANNLSS(
        family="normal",
        n_estimators=1,
        solver="native",
    )
    with pytest.raises(NotImplementedError, match="solver='native'"):
        native_only.fit(X, y, max_epochs=1)


def test_fitted_igann_persistence_preserves_native_state(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = X["x"] ** 2
    model = IGANNRegressor(
        n_hid=3,
        n_estimators=2,
        early_stopping=0,
        init_reg=0.01,
    ).fit(X, y, random_state=8)
    expected = model.predict(X)

    path = model.save_model(tmp_path / "igann.pkl")
    restored = IGANNRegressor.load_model(path)

    np.testing.assert_allclose(restored.predict(X), expected)
    assert restored.n_estimators_ == model.n_estimators_
    assert restored.training_history() == model.training_history()


@pytest.mark.skipif(
    importlib.util.find_spec("abess") is not None,
    reason="This check targets the optional-dependency error path.",
)
def test_igann_sparse_requires_abess_only_when_requested():
    model = IGANN(
        {},
        _numeric_info("x", "z"),
        config=DefaultIGANNConfig(n_estimators=1, sparse=1),
    )
    values = {
        "x": torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
        "z": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
    with pytest.raises(ImportError, match="abess"):
        model.fit_native(
            train_num_features=values,
            train_cat_features={},
            train_targets=torch.tensor([[0.0], [0.2], [0.8], [1.0]]),
            val_num_features=values,
            val_cat_features={},
            val_targets=torch.tensor([[0.0], [0.2], [0.8], [1.0]]),
            objective=BinaryObjective(),
        )


def test_igann_sparse_retains_at_most_requested_atomic_features():
    pytest.importorskip("abess")
    generator = np.random.default_rng(12)
    X = pd.DataFrame(generator.normal(size=(60, 3)), columns=["signal", "a", "b"])
    y = np.sin(2.0 * X["signal"]) + 0.02 * generator.normal(size=len(X))
    model = IGANNRegressor(
        n_hid=3,
        n_estimators=1,
        early_stopping=0,
        sparse=1,
        init_reg=0.01,
    )

    model.fit(X, y, random_state=5)

    assert 1 <= len(model.selected_features_) <= 1
    assert model.model_complexity()["selected_features"] == 1
