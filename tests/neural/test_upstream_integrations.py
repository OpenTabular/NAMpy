from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pretab.preprocessor import Preprocessor

from nampy.models._base import NeuralEstimatorBase
from nampy.models.nam import NAMLSS, NAMRegressor
from nampy.models.nbm import NBMRegressor
from nampy.neural.architectures.components.nam import ExU, NAMFeatureNN
from nampy.neural.architectures.components.oblivious_trees import ODST, ODSTBlock
from nampy.neural.architectures.components.sparse_activations import (
    sparsemax,
    sparsemoid,
)
from nampy.neural.architectures.components.transformer import (
    CustomTransformerEncoderLayer,
)
from nampy.neural.architectures.nam import NAM
from nampy.neural.architectures.nodegam import NodeGAM
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.configs.nodegam_config import DefaultNodeGAMConfig
from nampy.neural.distributions.distributions import Quantile
from nampy.neural.task import TaskModule


def test_exu_uses_per_unit_centering_exponential_slopes_and_relu_one_clipping():
    layer = ExU(1, 2)
    with torch.no_grad():
        layer.beta.copy_(torch.log(torch.tensor([[2.0, 4.0]])))
        layer.center.copy_(torch.tensor([[0.5, 0.75]]))
    inputs = torch.tensor([[0.25], [0.75], [2.0]])
    torch.testing.assert_close(
        layer(inputs),
        torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]]),
    )


def test_nam_exu_adaptive_width_and_regularizers_are_composable():
    model = NAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1, "n_unique": 4}},
        config=DefaultNAMConfig(
            feature_layer="exu",
            layer_sizes=[32, 8],
            adaptive_width=True,
            num_basis_functions=6,
            units_multiplier=2,
            feature_output_bias=False,
            output_regularization=0.25,
            l2_regularization=0.1,
            dropout=0.0,
        ),
    ).eval()
    feature_network = model.num_feature_networks["x"]
    assert isinstance(feature_network, NAMFeatureNN)
    assert feature_network.first_layer.out_features == 6
    assert feature_network.tail.linear_final.bias is None

    result = model({"x": torch.tensor([[0.0], [0.5], [1.0]])}, {})
    expected_output_penalty = 0.25 * torch.mean(result["x"].square())
    torch.testing.assert_close(result["output_regularizer"], expected_output_penalty)
    assert result["parameter_regularizer"].item() > 0


def test_nam_output_regularizer_uses_dropout_free_term_outputs():
    model = NAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1, "n_unique": 3}},
        config=DefaultNAMConfig(
            layer_sizes=[4],
            dropout=0.75,
            output_regularization=0.5,
        ),
    )
    inputs = {"x": torch.tensor([[0.0], [0.5], [1.0]])}
    model.train()
    training_result = model(inputs, {})
    model.eval()
    deterministic_result = model(inputs, {})
    expected = 0.5 * torch.mean(deterministic_result["x"].square())
    torch.testing.assert_close(training_result["output_regularizer"], expected)


def test_nam_exu_adaptive_width_fits_through_estimator_contract(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 36)})
    y = np.sin(X["x"].to_numpy())
    estimator = NAMRegressor(
        feature_layer="exu",
        layer_sizes=[16],
        adaptive_width=True,
        num_basis_functions=8,
        units_multiplier=2,
        output_regularization=1e-4,
        numerical_preprocessing="minmax",
    )
    estimator.fit(
        X,
        y,
        sample_weight=np.linspace(1.0, 2.0, len(y)),
        max_epochs=1,
        batch_size=12,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    feature_network = estimator.model.model.num_feature_networks["x"]
    assert feature_network.first_layer.out_features == 8
    components = estimator.predict_components(X, center=True)
    components.validate_additive_reconstruction()


def test_exu_is_not_accepted_as_a_conventional_activation():
    with pytest.raises(TypeError, match="feature_layer"):
        NAM(
            cat_feature_info={},
            num_feature_info={"x": {"dimension": 1}},
            config=DefaultNAMConfig(activation="exu"),
        )


def test_custom_transformer_matches_pytorch_norm_and_dropout_flow():
    torch.manual_seed(7)
    kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_first": True,
        "norm_first": True,
        "activation": F.relu,
    }
    reference = nn.TransformerEncoderLayer(**kwargs).eval()
    custom = CustomTransformerEncoderLayer(**kwargs).eval()
    custom.load_state_dict(reference.state_dict())

    source = torch.randn(3, 5, 8)
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    expected = reference(source, src_mask=mask, is_causal=True)
    actual = custom(source, src_mask=mask, is_causal=True)
    torch.testing.assert_close(actual, expected)


def test_sparsemax_and_sparsemoid_are_available_with_expected_geometry():
    logits = torch.tensor([[-2.0, 0.5, 3.0], [1.0, -1.0, 0.0]], requires_grad=True)
    probabilities = sparsemax(logits, dim=1)
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0
    )
    assert torch.all(probabilities >= 0)
    sparsemoid_values = sparsemoid(torch.tensor([-3.0, 0.0, 3.0]))
    torch.testing.assert_close(
        sparsemoid_values, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6, rtol=0
    )
    probabilities.sum().backward()
    assert logits.grad is not None


def test_nodegam_can_select_original_node_activations():
    model = NodeGAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        config=DefaultNodeGAMConfig(
            num_trees=4,
            num_layers=1,
            depth=2,
            selector_activation="sparsemax",
            bin_activation="sparsemoid",
            l2_lambda=0.1,
        ),
    ).eval()
    choices = [
        module.choice_function
        for module in model.model.modules()
        if hasattr(module, "choice_function")
    ]
    assert choices
    assert all(choice is sparsemax for choice in choices)
    result = model({"x": torch.tensor([[0.2], [0.8]])}, {})
    assert torch.isfinite(result["output"]).all()
    assert "output_penalty" in result


def test_quantile_preprocessing_uses_published_pretab_contract():
    X = pd.DataFrame({"x": np.repeat([0.0, 1.0, 2.0, 3.0], 8)})
    preprocessor = Preprocessor(
        numerical_preprocessing="quantile",
        scaling_strategy=None,
        treat_all_integers_as_numerical=True,
    ).fit(X)

    first = preprocessor.transform(X)["num_x"]
    second = preprocessor.transform(X)["num_x"]
    np.testing.assert_array_equal(first, second)


def test_nam_uses_published_pretab_grouped_block_metadata():
    X = pd.DataFrame(
        {
            "age": [20.0, 40.0, 30.0],
            "city": ["Berlin", "Paris", "Berlin"],
        }
    )
    estimator = NAMRegressor()
    estimator.preprocessor.fit(X, np.array([0.0, 1.0, 0.5]))

    transformed = estimator.preprocessor.transform(X)
    num_info, cat_info, _ = estimator.preprocessor.get_feature_info(verbose=False)

    assert list(transformed) == [
        "num_age",
        "cat_city",
    ]
    assert list(cat_info) == ["city"]
    assert list(num_info) == ["age"]
    assert cat_info["city"]["dimension"] == 2
    assert transformed["num_age"].shape[1] == 1
    assert transformed["cat_city"].shape[1] == 2

    model = NAM(
        cat_feature_info=cat_info,
        num_feature_info=num_info,
        config=DefaultNAMConfig(layer_sizes=[4], dropout=0.0),
    )
    assert model.feature_order == [
        ("num", "age"),
        ("cat", "city"),
    ]


def test_nbm_consumes_published_pretab_grouped_one_hot_blocks():
    X = pd.DataFrame({"group": ["a", "b", "c", "a"]})
    estimator = NBMRegressor()
    estimator.preprocessor.fit(X, np.array([0.0, 1.0, 0.5, 0.0]))

    transformed = estimator.preprocessor.transform(X)
    _, cat_info, _ = estimator.preprocessor.get_feature_info(verbose=False)

    assert list(transformed) == ["cat_group"]
    assert transformed["cat_group"].shape == (4, 3)
    assert cat_info["group"]["dimension"] == 3


def test_nodegam_forwards_interaction_specific_penalties():
    config = DefaultNodeGAMConfig(
        num_trees=4,
        num_layers=1,
        depth=2,
        interaction_degree=2,
        l2_interactions=0.25,
        l1_interactions=0.1,
        input_dropout=0.2,
    )
    model = NodeGAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}, "z": {"dimension": 1}},
        config=config,
    )
    assert model.model.l2_interactions == 0.25
    assert model.model.l1_interactions == 0.1
    assert model.model.input_dropout == 0.2


def test_odst_flatten_output_is_a_representation_option():
    flat = ODST(
        in_features=2,
        num_trees=3,
        depth=2,
        tree_dim=2,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
    ).eval()
    structured = ODST(
        in_features=2,
        num_trees=3,
        depth=2,
        tree_dim=2,
        flatten_output=False,
        choice_function=sparsemax,
        bin_function=sparsemoid,
    ).eval()
    X = torch.tensor([[0.1, 0.2], [0.7, -0.3]])
    flat_output = flat(X)
    structured.load_state_dict(flat.state_dict())
    structured_output = structured(X)
    assert flat_output.shape == (2, 6)
    assert structured_output.shape == (2, 3, 2)
    torch.testing.assert_close(flat_output, structured_output.flatten(1, 2))


def test_dense_odst_block_limits_layer_history_and_can_return_tree_axes():
    block = ODSTBlock(
        in_features=3,
        num_trees=2,
        num_layers=3,
        num_classes=1,
        max_features=4,
        flatten_output=False,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        add_last_linear=False,
    ).eval()
    assert [layer.in_features for layer in block] == [3, 4, 4]
    outputs = block.run_with_layers(torch.randn(5, 3))
    assert outputs.shape == (5, 6, 1)


def test_nodegam_masked_reconstruction_matches_reference_loss(monkeypatch):
    config = DefaultNodeGAMConfig(
        num_trees=2,
        num_layers=1,
        depth=2,
        interaction_degree=1,
        output_dropout=0.0,
        last_dropout=0.0,
    )
    task = TaskModule(
        model_class=NodeGAM,
        config=config,
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}, "z": {"dimension": 1}},
        num_classes=2,
        pretraining=True,
        pretraining_ratio=1.0,
        pretraining_noise=0.0,
    )
    captured = {}

    def fake_forward(num_features, cat_features, feature_masks=None):
        del cat_features
        captured["feature_masks"] = feature_masks
        targets = torch.cat(list(num_features.values()), dim=1)
        return {"output": torch.ones_like(targets)}

    monkeypatch.setattr(task.model, "forward", fake_forward)
    num = {"x": torch.tensor([[2.0], [0.0]]), "z": torch.tensor([[0.0], [2.0]])}
    loss = task._masked_reconstruction_step({}, num, stage="train")
    # Every feature is masked. The reference divides each row's squared
    # reconstruction errors by its number of masks, then averages all cells.
    assert loss.item() == 0.5
    torch.testing.assert_close(captured["feature_masks"], torch.ones(2, 2))


def test_nodegam_forward_exposes_learned_additive_terms_not_raw_inputs():
    model = NodeGAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}, "z": {"dimension": 1}},
        num_classes=1,
        config=DefaultNodeGAMConfig(
            num_trees=4,
            num_layers=1,
            depth=2,
            output_dropout=0.0,
            last_dropout=0.0,
        ),
    ).eval()
    features = {
        "x": torch.tensor([[0.1], [0.8]]),
        "z": torch.tensor([[0.7], [0.2]]),
    }
    result = model(num_features=features, cat_features={})

    assert result["x"].shape == features["x"].shape
    assert result["z"].shape == features["z"].shape
    assert not torch.equal(result["x"], features["x"])
    additive = result["intercept"].clone()
    for key, value in result.items():
        if key not in {"output", "intercept", "output_penalty"}:
            additive = additive + value
    torch.testing.assert_close(result["output"], additive)


def test_nodegam_temperature_callback_reaches_selector_functions():
    model = NodeGAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        config=DefaultNodeGAMConfig(anneal_steps=4),
    )
    model.temp_step_callback(4)
    taus = [
        module.choice_function.tau.item()
        for module in model.model.modules()
        if hasattr(getattr(module, "choice_function", None), "tau")
    ]
    assert taus
    assert all(abs(tau - 0.01) < 1e-7 for tau in taus)


def test_qnam_output_is_not_transformed_twice():
    from nampy.neural.architectures.qnam import QNAM
    from nampy.neural.configs.qnam_config import DefaultQNAMConfig

    model = QNAM(
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}},
        num_classes=3,
        config=DefaultQNAMConfig(layer_sizes=[4], dropout=0.0),
    ).eval()
    output = model({"x": torch.tensor([[0.2], [0.8]])}, {})["output"]
    family_output = Quantile(enforce_monotonic=False)(output)
    torch.testing.assert_close(family_output, output)
    assert torch.all(output[:, 1:] >= output[:, :-1])


def test_lss_infers_ordinal_and_multivariate_dimensions(monkeypatch):
    def skip_training(self, X, y, **kwargs):
        return self

    monkeypatch.setattr(NeuralEstimatorBase, "fit", skip_training)

    ordinal = NAMLSS(family="ordinal")
    ordinal.fit([[0.0], [1.0], [2.0]], [0, 1, 2])
    assert ordinal.family_.num_classes == 3
    assert ordinal.family_.param_count == 1

    mvnormal = NAMLSS(family="mvnormdiag")
    mvnormal.fit([[0.0], [1.0]], [[0.0, 1.0], [1.0, 2.0]])
    assert mvnormal.family_.n_dim == 2
    assert mvnormal.family_.param_count == 4
