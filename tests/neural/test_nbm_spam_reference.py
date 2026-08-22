"""Static parity checks against the NBM-SPAM reference implementation."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from nampy.models.nbm import NBMRegressor
from nampy.neural.architectures.nbm import NBM
from nampy.neural.architectures.nbm_spam import NBMSPAM
from nampy.neural.architectures.spam import SPAM
from nampy.neural.configs.nbm_config import DefaultNBMConfig
from nampy.neural.configs.nbm_spam_config import DefaultNBMSPAMConfig
from nampy.neural.configs.spam_config import DefaultSPAMConfig
from tests.reference_fixtures import load_reference, reference_key, save_reference


@lru_cache(maxsize=1)
def _upstream_modules():
    root = Path(__file__).resolve().parents[2] / "upstreams" / "nbm-spam"
    registry_module = ModuleType("fvcore.common.registry")

    class Registry:
        def __init__(self, _name):
            self._objects = {}

        def register(self, value=None):
            def decorator(item):
                self._objects[item.__name__] = item
                return item

            return decorator(value) if value is not None else decorator

        def get(self, name):
            return self._objects[name]

    registry_module.Registry = Registry
    fvcore_module = ModuleType("fvcore")
    common_module = ModuleType("fvcore.common")
    stubs = {
        "fvcore": fvcore_module,
        "fvcore.common": common_module,
        "fvcore.common.registry": registry_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stubs}
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)
    sys.path.insert(0, str(root))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        concept_nbm = importlib.import_module("nbm_spam.models.concept_nbm")
        concept_spam = importlib.import_module("nbm_spam.models.concept_spam")
    finally:
        sys.dont_write_bytecode = previous
        sys.path.remove(str(root))
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
    return concept_nbm, concept_spam


def _info(count: int):
    return {f"x{index}": {"dimension": 1} for index in range(count)}


def _inputs(matrix: torch.Tensor):
    return {
        f"x{index}": matrix[:, index : index + 1]
        for index in range(matrix.shape[1])
    }


def _copy_parameters(source, target) -> None:
    source_parameters = list(source.parameters())
    target_parameters = list(target.parameters())
    assert [parameter.shape for parameter in source_parameters] == [
        parameter.shape for parameter in target_parameters
    ]
    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            source_parameters, target_parameters, strict=True
        ):
            target_parameter.copy_(source_parameter)


def _encode_parameters(module) -> list:
    return [parameter.detach().cpu().tolist() for parameter in module.parameters()]


def _load_parameters(module, values) -> None:
    parameters = list(module.parameters())
    assert len(parameters) == len(values)
    with torch.no_grad():
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))


def _encode_state(module) -> dict:
    return {
        name: value.detach().cpu().tolist()
        for name, value in module.state_dict().items()
    }


def _load_state(module, values: dict) -> None:
    current = module.state_dict()
    module.load_state_dict(
        {
            name: torch.as_tensor(value, dtype=current[name].dtype)
            for name, value in values.items()
        }
    )


def _nbm_spam_fixture(operation: str, payload: dict, producer):
    key = reference_key(operation, payload)
    fixture = load_reference("nbm_spam", key)
    if fixture is None:
        fixture = producer()
        save_reference("nbm_spam", key, fixture)
    return fixture


def test_nbm_defaults_use_pristine_pretab_block_contract():
    config = DefaultNBMConfig()
    assert config.layer_sizes == [256, 128, 128]
    assert config.dropout == 0.0
    assert config.bases_dropout == 0.0
    assert config.batch_norm is True
    assert config.num_bases == 100
    assert config.num_subnets == 1
    assert config.featurizer == "conv1d"
    assert config.sparse is False

    estimator = NBMRegressor()
    params = estimator.get_params(deep=False)
    assert params["numerical_method"] == "none"
    assert params["categorical_method"] == "one-hot"
    assert params["scaling"] == "minmax"
    assert params["dtype"] is np.float32


def test_nbm_flattens_pristine_pretab_grouped_blocks_to_atomic_concepts():
    model = NBM(
        cat_feature_info={
            "group": {"dimension": 2},
        },
        num_feature_info={"x": {"dimension": 1}},
        config=DefaultNBMConfig(
            layer_sizes=[4],
            num_bases=3,
            batch_norm=False,
            nary=[1],
        ),
    )

    assert model.feature_order == [
        ("num", "x"),
        ("cat", "group"),
    ]
    assert model.atomic_feature_names == ["x", "group[0]", "group[1]"]


def test_dense_nbm_forward_features_penalty_and_gradients_match_upstream():
    matrix_values = [[0.1, 0.4, 0.8], [0.7, 0.2, 0.3], [0.9, 0.6, 0.5]]

    def produce():
        concept_nbm, _ = _upstream_modules()
        upstream = concept_nbm.ConceptNBMNary(
            num_concepts=3,
            num_classes=2,
            nary=[1, 2],
            num_bases=4,
            hidden_dims=(5,),
            num_subnets=2,
            dropout=0.0,
            bases_dropout=0.0,
            batchnorm=False,
            output_penalty=0.3,
        )
        matrix = torch.tensor(matrix_values, requires_grad=True)
        output, features = upstream(matrix)
        gradient = torch.autograd.grad(output.sum(), matrix)[0]
        return {
            "bases": {
                key: _encode_parameters(module)
                for key, module in upstream.bases_nary_models.items()
            },
            "featurizer": _encode_state(upstream.featurizer),
            "classifier": _encode_state(upstream.classifier),
            "output": output.detach().cpu().tolist(),
            "features": features.detach().cpu().tolist(),
            "gradient": gradient.detach().cpu().tolist(),
        }

    fixture = _nbm_spam_fixture(
        "dense_nbm",
        {"num_concepts": 3, "num_classes": 2, "nary": [1, 2], "seed": 0},
        produce,
    )
    model = NBM(
        {},
        _info(3),
        num_classes=2,
        config=DefaultNBMConfig(
            nary=[1, 2],
            num_bases=4,
            layer_sizes=[5],
            num_subnets=2,
            dropout=0.0,
            bases_dropout=0.0,
            batch_norm=False,
            output_penalty=0.3,
        ),
    )
    for key, values in fixture["bases"].items():
        _load_parameters(model.core.bases_nary_models[key], values)
    assert model.core.featurizer is not None
    _load_state(model.core.featurizer, fixture["featurizer"])
    _load_state(model.classifier, fixture["classifier"])

    matrix = torch.tensor(matrix_values, requires_grad=True)
    upstream_output = torch.tensor(fixture["output"])
    upstream_features = torch.tensor(fixture["features"])
    result = model(_inputs(matrix), {})

    torch.testing.assert_close(model.core(matrix), upstream_features)
    torch.testing.assert_close(result["output"], upstream_output)
    torch.testing.assert_close(
        result["output_penalty"], 0.3 * upstream_features.square().mean()
    )
    reconstruction = result["intercept"] + sum(
        value
        for key, value in result.items()
        if key not in {"output", "intercept", "output_penalty"}
    )
    torch.testing.assert_close(result["output"], reconstruction)

    model_gradient = torch.autograd.grad(result["output"].sum(), matrix)[0]
    torch.testing.assert_close(model_gradient, torch.tensor(fixture["gradient"]))

    model.eval()
    assert "output_penalty" not in model(_inputs(matrix.detach()), {})


def test_einsum_and_upstream_conv1d_featurizers_are_equivalent():
    conv_model = NBM(
        {},
        _info(2),
        config=DefaultNBMConfig(
            layer_sizes=[4], num_bases=3, batch_norm=False, featurizer="conv1d"
        ),
    ).eval()
    einsum_model = NBM(
        {},
        _info(2),
        config=DefaultNBMConfig(
            layer_sizes=[4], num_bases=3, batch_norm=False, featurizer="einsum"
        ),
    ).eval()
    for key in conv_model.core.bases_nary_models:
        _copy_parameters(
            conv_model.core.bases_nary_models[key],
            einsum_model.core.bases_nary_models[key],
        )
    conv = conv_model.core.featurizer
    assert conv is not None
    with torch.no_grad():
        einsum_model.core.featurizer_weight.copy_(conv.weight.squeeze(-1))
        einsum_model.core.featurizer_bias.copy_(conv.bias)
    einsum_model.classifier.load_state_dict(conv_model.classifier.state_dict())

    matrix = torch.tensor([[0.2, 0.7], [0.8, 0.1]])
    torch.testing.assert_close(
        conv_model(_inputs(matrix), {})["output"],
        einsum_model(_inputs(matrix), {})["output"],
    )


def test_sparse_nbm_active_tuple_path_matches_upstream():
    matrix_values = [[0.0, 0.4, 0.0], [0.7, 0.0, 0.3], [0.0, 0.0, 0.0]]

    def produce():
        concept_nbm, _ = _upstream_modules()
        upstream = concept_nbm.ConceptNBMNarySparse(
            num_concepts=3,
            num_classes=2,
            nary=[1, 2],
            num_bases=3,
            hidden_dims=(4,),
            dropout=0.0,
            bases_dropout=0.0,
            batchnorm=False,
            output_penalty=0.2,
            nary_ignore_input=0.0,
        )
        output, features = upstream(torch.tensor(matrix_values))
        return {
            "bases": {
                str(order): _encode_parameters(module)
                for order, module in upstream.bases_nary_models.items()
            },
            "featurizers": {
                str(order): {
                    "weight": values["weight"].detach().cpu().tolist(),
                    "bias": values["bias"].detach().cpu().tolist(),
                }
                for order, values in upstream.featurizer_params.items()
            },
            "classifier": _encode_state(upstream.classifier),
            "output": output.detach().cpu().tolist(),
            "features": features.detach().cpu().tolist(),
        }

    fixture = _nbm_spam_fixture(
        "sparse_nbm",
        {"num_concepts": 3, "num_classes": 2, "nary": [1, 2], "seed": 0},
        produce,
    )
    model = NBM(
        {},
        _info(3),
        num_classes=2,
        config=DefaultNBMConfig(
            nary=[1, 2],
            num_bases=3,
            layer_sizes=[4],
            dropout=0.0,
            bases_dropout=0.0,
            batch_norm=False,
            output_penalty=0.2,
            sparse=True,
            nary_ignore_input=0.0,
        ),
    )
    for order_text, values in fixture["bases"].items():
        order = int(order_text)
        _load_parameters(model.core.bases_nary_models[f"ord{order}_net0"], values)
        with torch.no_grad():
            model.core.featurizer_params[order_text]["weight"].copy_(
                torch.tensor(fixture["featurizers"][order_text]["weight"])
            )
            model.core.featurizer_params[order_text]["bias"].copy_(
                torch.tensor(fixture["featurizers"][order_text]["bias"])
            )
    _load_state(model.classifier, fixture["classifier"])

    matrix = torch.tensor(matrix_values)
    upstream_output = torch.tensor(fixture["output"])
    upstream_features = torch.tensor(fixture["features"])
    result = model(_inputs(matrix), {})
    torch.testing.assert_close(model.core(matrix), upstream_features)
    torch.testing.assert_close(result["output"], upstream_output)
    assert torch.count_nonzero(model.core(matrix)[-1]) == 0


def test_sparse_nbm_defines_all_ignored_batches_and_rejects_subnet_ambiguity():
    model = NBM(
        {},
        _info(3),
        config=DefaultNBMConfig(
            nary=[1],
            num_bases=3,
            layer_sizes=[4],
            batch_norm=False,
            sparse=True,
            nary_ignore_input=-1.0,
        ),
    ).eval()
    matrix = torch.full((4, 3), -1.0)
    assert torch.count_nonzero(model.core(matrix)) == 0

    with pytest.raises(ValueError, match="num_subnets=1"):
        NBM(
            {},
            _info(2),
            config=DefaultNBMConfig(sparse=True, num_subnets=2),
        )


def test_spam_forward_and_regularizers_match_upstream():
    matrix_values = [[0.1, 0.4, 0.8], [0.7, 0.2, 0.3], [0.9, 0.6, 0.5]]

    def produce():
        _, concept_spam = _upstream_modules()
        upstream = concept_spam.ConceptSPAM(
            num_concepts=3,
            num_classes=2,
            ranks=[3, 2],
            dropout=0.0,
            reg_order=2,
            lower_order_correction=True,
            use_geometric_mean=True,
        )
        return {
            "poly_weights": [_encode_state(module) for module in upstream.poly_weights],
            "classifier": _encode_state(upstream.classifier),
            "output": upstream(torch.tensor(matrix_values)).detach().cpu().tolist(),
            "tensor_regularization": float(upstream.tensor_regularization().detach()),
            "basis_l1_regularization": float(
                upstream.basis_l1_regularization().detach()
            ),
        }

    fixture = _nbm_spam_fixture(
        "spam_forward",
        {"num_concepts": 3, "num_classes": 2, "ranks": [3, 2], "seed": 0},
        produce,
    )
    model = SPAM(
        {},
        _info(3),
        num_classes=2,
        config=DefaultSPAMConfig(
            ranks=[3, 2],
            dropout=0.0,
            reg_order=2,
            lower_order_correction=True,
            use_geometric_mean=True,
            regularization_scale=0.4,
            basis_l1_regularization=0.2,
        ),
    )
    for source, target in zip(fixture["poly_weights"], model.core.poly_weights, strict=True):
        _load_state(target, source)
    _load_state(model.core.classifier, fixture["classifier"])

    matrix = torch.tensor(matrix_values)
    result = model(_inputs(matrix), {})
    torch.testing.assert_close(result["output"], torch.tensor(fixture["output"]))
    torch.testing.assert_close(
        result["tensor_regularizer"],
        torch.tensor(0.4 * fixture["tensor_regularization"]),
    )
    torch.testing.assert_close(
        result["basis_l1_regularizer"],
        torch.tensor(0.2 * fixture["basis_l1_regularization"]),
    )


def test_spam_quadratic_local_importance_matches_upstream():
    row = torch.tensor([0.2, 0.5, 0.9])

    def produce():
        _, concept_spam = _upstream_modules()
        upstream = concept_spam.ConceptSPAM(
            num_concepts=3, num_classes=1, ranks=[4], dropout=0.0
        ).eval()
        upstream_index, upstream_value = upstream.get_importance(
            row, target=0, top_k=1
        )[0]
        upstream_term = (
            (upstream_index,)
            if upstream_index < 3
            else list(combinations(range(3), 2))[upstream_index - 3]
        )
        return {
            "poly_weights": [_encode_state(module) for module in upstream.poly_weights],
            "classifier": _encode_state(upstream.classifier),
            "term": list(upstream_term),
            "value": float(upstream_value),
        }

    fixture = _nbm_spam_fixture(
        "spam_local_importance",
        {"num_concepts": 3, "num_classes": 1, "ranks": [4], "seed": 0},
        produce,
    )
    model = SPAM(
        {},
        _info(3),
        config=DefaultSPAMConfig(ranks=[4], dropout=0.0),
    ).eval()
    for source, target in zip(fixture["poly_weights"], model.core.poly_weights, strict=True):
        _load_state(target, source)
    _load_state(model.core.classifier, fixture["classifier"])

    actual_term, actual_value = model.core.local_term_importance(
        row.unsqueeze(0), target=0, top_k=1
    )[0][0]

    assert actual_term == tuple(fixture["term"])
    assert actual_value == pytest.approx(fixture["value"])


def test_spam_proximal_projection_and_training_only_penalties():
    model = SPAM(
        {},
        _info(2),
        config=DefaultSPAMConfig(
            ranks=[3],
            proximal=True,
            regularization_scale=0.2,
            basis_l1_regularization=0.1,
        ),
    )
    with torch.no_grad():
        model.core.classifier.weight.fill_(-1.0)
        model.core.poly_weights[0].weight.fill_(-1.0)
    matrix = torch.tensor([[0.2, 0.8], [0.5, 0.4]])
    result = model(_inputs(matrix), {})
    assert torch.all(model.core.classifier.weight >= 0)
    assert torch.all(model.core.poly_weights[0].weight >= 0)
    assert "tensor_regularizer" in result
    assert "basis_l1_regularizer" in result

    model.eval()
    result = model(_inputs(matrix), {})
    assert "tensor_regularizer" not in result
    assert "basis_l1_regularizer" not in result


def test_nbm_spam_hybrid_matches_upstream_block_assembly():
    polynomial = {"ranks": [3], "dropout": 0.0}
    matrix_values = [[0.1, 0.4, 0.8], [0.7, 0.2, 0.3], [0.9, 0.6, 0.5]]

    def produce():
        concept_nbm, _ = _upstream_modules()
        upstream = concept_nbm.ConceptNBMNary(
            num_concepts=3,
            num_classes=2,
            nary=None,
            num_bases=4,
            hidden_dims=(5,),
            num_subnets=1,
            dropout=0.0,
            bases_dropout=0.0,
            batchnorm=False,
            output_penalty=0.2,
            polynomial=polynomial,
        )
        output, features = upstream(torch.tensor(matrix_values))
        return {
            "bases": {
                key: _encode_parameters(module)
                for key, module in upstream.bases_nary_models.items()
            },
            "featurizer": _encode_state(upstream.featurizer),
            "linear_head": _encode_state(upstream._spam[0]),
            "polynomial_head": _encode_parameters(upstream._spam[1]),
            "output": output.detach().cpu().tolist(),
            "features": features.detach().cpu().tolist(),
        }

    fixture = _nbm_spam_fixture(
        "nbm_spam_hybrid",
        {"num_concepts": 3, "num_classes": 2, "ranks": [3], "seed": 0},
        produce,
    )
    model = NBMSPAM(
        {},
        _info(3),
        num_classes=2,
        config=DefaultNBMSPAMConfig(
            ranks=[3],
            num_bases=4,
            layer_sizes=[5],
            num_subnets=1,
            dropout=0.0,
            bases_dropout=0.0,
            batch_norm=False,
            output_penalty=0.2,
        ),
    )
    for key, values in fixture["bases"].items():
        _load_parameters(model.core.bases_nary_models[key], values)
    _load_state(model.core.featurizer, fixture["featurizer"])
    _load_state(model.linear_head, fixture["linear_head"])
    _load_parameters(model.polynomial_heads[0], fixture["polynomial_head"])

    matrix = torch.tensor(matrix_values)
    upstream_output = torch.tensor(fixture["output"])
    upstream_features = torch.tensor(fixture["features"])
    result = model(_inputs(matrix), {})
    torch.testing.assert_close(model.core(matrix), upstream_features)
    torch.testing.assert_close(result["output"], upstream_output)
    reconstruction = result["intercept"] + sum(
        value
        for key, value in result.items()
        if key not in {"output", "intercept", "output_penalty"}
    )
    torch.testing.assert_close(result["output"], reconstruction)


def test_nbm_spam_rejects_preexpanded_nary_terms():
    with pytest.raises(ValueError, match="unary NBM terms"):
        NBMSPAM(
            {},
            _info(3),
            config=DefaultNBMSPAMConfig(ranks=[3]),
            nary=[1, 2],
        )
