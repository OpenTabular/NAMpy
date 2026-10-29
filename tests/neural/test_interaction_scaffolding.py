"""The shared interaction scaffolding preserves key grammar and term-mask dropout."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nampy.neural.architectures.components.interactions import (
    apply_feature_dropout,
    create_interaction_networks,
    interaction_forward,
    sum_feature_dims,
)


def test_create_interaction_networks_key_grammar():
    names = ["a", "b", "c"]
    networks = create_interaction_networks(
        names, 3, lambda interaction: nn.Linear(len(interaction), 1)
    )
    # ":"-joined keys in combinations order, degrees 2 then 3.
    assert list(networks.keys()) == ["a:b", "a:c", "b:c", "a:b:c"]


def test_create_interaction_networks_accepts_an_explicit_sparse_set():
    networks = create_interaction_networks(
        ["a", "b", "c"],
        None,
        lambda interaction: nn.Linear(len(interaction), 1),
        interactions=(("c", "a"),),
    )
    assert list(networks) == ["a:c"]

    with pytest.raises(ValueError, match="either interactions"):
        create_interaction_networks(
            ["a", "b", "c"],
            2,
            lambda interaction: nn.Linear(len(interaction), 1),
            interactions=(("a", "c"),),
        )


@pytest.mark.parametrize("degree", [None, 0, 1])
def test_create_interaction_networks_disabled(degree):
    networks = create_interaction_networks(
        ["a", "b"], degree, lambda interaction: nn.Linear(2, 1)
    )
    assert len(networks) == 0


def test_sum_feature_dims_mixed_feature_infos():
    num_info = {"x": {"dimension": 2}}
    cat_info = {"c": {"dimension": 3}}
    assert sum_feature_dims(("x", "c"), num_info, cat_info) == 5


def test_interaction_forward_concatenates_members_in_key_order():
    networks = create_interaction_networks(
        ["x", "z"], 2, lambda interaction: nn.Identity()
    )
    num_features = {"x": torch.tensor([[1.0], [2.0]])}
    cat_features = {"z": torch.tensor([[3.0], [4.0]])}
    outputs = interaction_forward(
        networks,
        2,
        num_features,
        cat_features,
        lambda network, input_features: network(input_features),
    )
    assert set(outputs) == {"x:z"}
    assert torch.equal(outputs["x:z"], torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_apply_feature_dropout_masks_whole_terms():
    torch.manual_seed(7)
    term_outputs = torch.ones(64, 5, 3)
    dropped = apply_feature_dropout(term_outputs, 0.5, training=True)

    # Each (sample, term) is either fully zero or fully scaled by 1/(1-p)
    # across the output dimension.
    per_term = dropped[:, :, 0]
    assert torch.all((per_term == 0.0) | torch.isclose(per_term, torch.tensor(2.0)))
    assert torch.equal(dropped, per_term.unsqueeze(-1).expand_as(dropped))
    assert (per_term == 0.0).any()
    assert (per_term != 0.0).any()


def test_apply_feature_dropout_identity_when_inactive():
    term_outputs = torch.randn(4, 3, 2)
    assert torch.equal(
        apply_feature_dropout(term_outputs, 0.0, training=True), term_outputs
    )
    assert torch.equal(
        apply_feature_dropout(term_outputs, 0.5, training=False), term_outputs
    )


@pytest.mark.parametrize(
    "name", ["nam", "qnam", "treenam", "spline_nam", "natt", "namformer"]
)
def test_registered_feature_dropout_preserves_forward_decomposition(name):
    from dataclasses import fields, replace

    from nampy.neural.registry import get_architecture

    spec = get_architecture(name)
    config = spec.config()
    options = {
        "feature_dropout": 0.5,
        "dropout": 0.0,
        "head_dropout": 0.0,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "head_layer_sizes": [4],
        "layer_sizes": [4],
    }
    config = replace(
        config,
        **{
            key: value
            for key, value in options.items()
            if key in {field.name for field in fields(config)}
        },
    )
    outputs = 1 if name == "spline_nam" else 3
    model = spec.module(
        config=config,
        cat_feature_info={},
        num_feature_info={"x": {"dimension": 1}, "z": {"dimension": 1}},
        num_classes=outputs,
    )
    inputs = {"x": torch.randn(64, 1), "z": torch.randn(64, 1)}
    model.train()
    result = model(inputs, {})
    terms = [
        value
        for key, value in result.items()
        if key not in {"output", "intercept"} and not key.endswith("penalty")
    ]
    reconstructed = torch.stack(terms).sum(dim=0) + result.get("intercept", 0.0)
    torch.testing.assert_close(result["output"], reconstructed)
    for term in terms:
        zeros = term == 0
        assert torch.equal(zeros, zeros[:, :1].expand_as(zeros))
    assert any((term == 0).all(dim=1).any() for term in terms)
    model.eval()
    with_dropout_setting = model(inputs, {})["output"]
    model.feature_dropout_p = 0.0
    torch.testing.assert_close(with_dropout_setting, model(inputs, {})["output"])
