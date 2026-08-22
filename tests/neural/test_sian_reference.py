from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn

from nampy.neural.architectures.nam import NAM
from nampy.neural.architectures.sian import SIAN, MuPResidualMLP
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.configs.sian_config import DefaultSIANConfig
from tests.reference_fixtures import load_reference, reference_key, save_reference


def _load_upstream_sian_models() -> ModuleType:
    models_path = (
        Path(__file__).resolve().parents[2]
        / "upstreams"
        / "sian"
        / "src"
        / "sian"
        / "models"
        / "models.py"
    )
    spec = importlib.util.spec_from_file_location("upstream_sian_models", models_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def _copy_sian_terms_to_nampy(upstream, model: NAM, term_names: list[str]) -> None:
    networks = [*model.num_feature_networks.values(), *model.interaction_networks.values()]
    assert len(networks) == len(term_names) == len(upstream.all_indices)

    source_input_offset = 0
    for layer_index, source_layer in enumerate(upstream.hiddens):
        source_output_offset = 0
        for term_index, (indices, network) in enumerate(
            zip(upstream.all_indices, networks, strict=True)
        ):
            output_width = upstream.all_sizes[term_index][layer_index + 1]
            output_slice = slice(
                source_output_offset, source_output_offset + output_width
            )
            target_layer = (
                network.hidden_layers[layer_index].block[0]
                if layer_index < len(network.hidden_layers)
                else network.linear_final
            )

            with torch.no_grad():
                if layer_index == 0:
                    target_layer.weight.copy_(source_layer.weight[output_slice][:, indices])
                else:
                    input_width = upstream.all_sizes[term_index][layer_index]
                    input_slice = slice(
                        source_input_offset, source_input_offset + input_width
                    )
                    target_layer.weight.copy_(
                        source_layer.weight[output_slice, input_slice]
                    )
                target_layer.bias.copy_(source_layer.bias[output_slice])

            source_output_offset += output_width
            if layer_index > 0:
                source_input_offset += upstream.all_sizes[term_index][layer_index]
        source_input_offset = 0

    with torch.no_grad():
        model.intercept.copy_(upstream.bias)


def _tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _encode_layers(layers) -> list[dict]:
    return [
        {
            "weight": layer.weight.detach().cpu().tolist(),
            "bias": layer.bias.detach().cpu().tolist(),
        }
        for layer in layers
    ]


def _decode_block_reference(fixture: dict) -> SimpleNamespace:
    return SimpleNamespace(
        all_indices=[tuple(indices) for indices in fixture["all_indices"]],
        all_sizes=fixture["all_sizes"],
        hiddens=[
            SimpleNamespace(weight=_tensor(layer["weight"]), bias=_tensor(layer["bias"]))
            for layer in fixture["hiddens"]
        ],
        bias=_tensor(fixture["bias"]),
    )


def test_nampy_sparse_higher_order_terms_match_upstream_sian():
    feature_names = ["x0", "x1", "x2", "x3"]
    indices = [(0,), (1,), (2,), (3,), (1, 2), (0, 2, 3)]
    term_names = ["x0", "x1", "x2", "x3", "x1:x2", "x0:x2:x3"]
    inputs = torch.tensor(
        [
            [-1.5, 0.2, 0.7, 1.1],
            [0.5, -0.8, 1.2, -0.4],
            [1.3, 0.9, -1.1, 0.6],
            [-0.2, 1.7, 0.3, -1.4],
        ]
    )
    key = reference_key(
        "sparse_higher_order_terms",
        {"seed": 91, "indices": indices, "small_sizes": [0, 4, 3, 1]},
    )
    fixture = load_reference("sian", key)
    if fixture is None:
        upstream_models = _load_upstream_sian_models()
        torch.manual_seed(91)
        source = upstream_models.Blocksparse_Deep_Relu_GAM(
            feat_in=len(feature_names),
            all_indices=indices,
            small_sizes=[0, 4, 3, 1],
        ).eval()
        source_output, _ = source(inputs)
        source_terms = source.forward_shapes(inputs)
        fixture = {
            "all_indices": [list(item) for item in source.all_indices],
            "all_sizes": source.all_sizes,
            "hiddens": _encode_layers(source.hiddens),
            "bias": source.bias.detach().cpu().tolist(),
            "output": source_output.detach().cpu().tolist(),
            "terms": source_terms.detach().cpu().tolist(),
        }
        save_reference("sian", key, fixture)
    upstream = _decode_block_reference(fixture)
    model = NAM(
        cat_feature_info={},
        num_feature_info={name: {"dimension": 1} for name in feature_names},
        config=DefaultNAMConfig(
            layer_sizes=[4, 3],
            activation=nn.ReLU,
            dropout=0.0,
            interactions=[("x1", "x2"), ("x0", "x2", "x3")],
            feature_output_bias=True,
            intercept=True,
        ),
    ).eval()
    _copy_sian_terms_to_nampy(upstream, model, term_names)
    upstream_output = _tensor(fixture["output"])
    upstream_terms = _tensor(fixture["terms"])
    num_inputs = {
        name: inputs[:, position : position + 1]
        for position, name in enumerate(feature_names)
    }
    result = model(num_inputs, {})
    nampy_terms = torch.cat([result[name] for name in term_names], dim=1)

    torch.testing.assert_close(nampy_terms, upstream_terms)
    torch.testing.assert_close(result["output"], upstream_output)
    torch.testing.assert_close(
        result["output"], nampy_terms.sum(dim=1, keepdim=True) + result["intercept"]
    )



def test_nampy_sian_block_representation_matches_upstream_sian():
    indices = [(0,), (1,), (2,), (3,), (1, 2), (0, 2, 3)]
    torch.manual_seed(103)
    inputs = torch.randn(8, 4)
    key = reference_key(
        "block_representation",
        {"seed": 103, "indices": indices, "small_sizes": [0, 4, 3, 1]},
    )
    fixture = load_reference("sian", key)
    if fixture is None:
        upstream_models = _load_upstream_sian_models()
        torch.manual_seed(103)
        source = upstream_models.Blocksparse_Deep_Relu_GAM(
            feat_in=4,
            all_indices=indices,
            small_sizes=[0, 4, 3, 1],
        ).eval()
        source_output, _ = source(inputs)
        fixture = {
            "hiddens": _encode_layers(source.hiddens),
            "output": source_output.detach().cpu().tolist(),
            "terms": source.forward_shapes(inputs).detach().cpu().tolist(),
        }
        save_reference("sian", key, fixture)
    torch.manual_seed(103)
    model = SIAN(
        cat_feature_info={},
        num_feature_info={f"x{index}": {"dimension": 1} for index in range(4)},
        config=DefaultSIANConfig(
            layer_sizes=[4, 3],
            interactions=[("x1", "x2"), ("x0", "x2", "x3")],
            l1_regularization=0.0,
            execution_mode="block_masked",
        ),
    ).eval()
    assert model.block_network is not None
    for source, target in zip(
        fixture["hiddens"], model.block_network.layers, strict=True
    ):
        torch.testing.assert_close(target.weight, _tensor(source["weight"]), rtol=0, atol=0)
        torch.testing.assert_close(target.bias, _tensor(source["bias"]), rtol=0, atol=0)

    upstream_output = _tensor(fixture["output"])
    upstream_terms = _tensor(fixture["terms"])
    result = model(
        {f"x{index}": inputs[:, index : index + 1] for index in range(4)}, {}
    )
    nampy_terms = torch.cat([result[name] for name in model.term_names_], dim=1)
    torch.testing.assert_close(nampy_terms, upstream_terms)
    torch.testing.assert_close(result["output"], upstream_output)

    model.compress()
    assert model.execution_mode_ == "independent"
    compressed = model(
        {f"x{index}": inputs[:, index : index + 1] for index in range(4)}, {}
    )
    torch.testing.assert_close(compressed["output"], upstream_output)
    model.block_mask()
    assert model.execution_mode_ == "block_masked"
    torch.testing.assert_close(
        model(
            {f"x{index}": inputs[:, index : index + 1] for index in range(4)}, {}
        )["output"],
        upstream_output,
    )


def test_nampy_sian_optional_residual_matches_upstream_mup_network():
    torch.manual_seed(211)
    inputs = torch.randn(9, 3)
    key = reference_key(
        "optional_residual", {"seed": 211, "layer_sizes": [3, 5, 4, 1]}
    )
    fixture = load_reference("sian", key)
    if fixture is None:
        upstream_models = _load_upstream_sian_models()
        torch.manual_seed(211)
        source = upstream_models.MuP_Relu_DNN([3, 5, 4, 1]).eval()
        fixture = {
            "hiddens": _encode_layers(source.hiddens),
            "output": source(inputs).detach().cpu().tolist(),
        }
        save_reference("sian", key, fixture)
    torch.manual_seed(211)
    residual = MuPResidualMLP(3, [5, 4], 1).eval()

    for source, target in zip(fixture["hiddens"], residual.layers, strict=True):
        torch.testing.assert_close(target.weight, _tensor(source["weight"]), rtol=0, atol=0)
        torch.testing.assert_close(target.bias, _tensor(source["bias"]), rtol=0, atol=0)
    torch.testing.assert_close(residual(inputs), _tensor(fixture["output"]))
