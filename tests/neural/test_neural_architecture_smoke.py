from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from nampy.neural.modules.ensemble_treenam import EnsembleTreeNAM
from nampy.neural.modules.gpnam import GPNAM
from nampy.neural.modules.linreg import LinReg
from nampy.neural.modules.multi_model import MultiModelWrapper
from nampy.neural.modules.nam import NAM
from nampy.neural.modules.namformer import NAMformer
from nampy.neural.modules.natt import NATT
from nampy.neural.modules.nbm import NBM
from nampy.neural.modules.nodegam import NodeGAM
from nampy.neural.modules.qnam import QNAMBase
from nampy.neural.modules.snam import SNAM
from nampy.neural.modules.spline_nam import SplineNAM
from nampy.neural.modules.treenam import TreeNAM
from nampy.neural.configs.ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from nampy.neural.configs.gpnam_config import DefaultGPNAMConfig
from nampy.neural.configs.linreg_config import DefaultLinRegConfig
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.configs.namformer_config import DefaultNAMformerConfig
from nampy.neural.configs.natt_config import DefaultNATTConfig
from nampy.neural.configs.nbm_config import DefaultNBMConfig
from nampy.neural.configs.nodegam_config import DefaultNodeGAMConfig
from nampy.neural.configs.qnam_config import DefaultQNAMConfig
from nampy.neural.configs.snam_config import DefaultSNAMConfig
from nampy.neural.configs.spline_nam_config import DefaultSplineNAMConfig
from nampy.neural.configs.treenam_config import DefaultTreeNAMConfig


@dataclass(frozen=True)
class _ArchitectureCase:
    name: str
    model_class: type
    config: object
    output_dim: int = 2
    expects_penalty: bool = False
    monotone_output: bool = False
    expected_keys: tuple[str, ...] = ()


class _PenalizedScalarModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, num_features, cat_features):
        output = next(iter(num_features.values())) * self.weight
        return {
            "output": output,
            "toy_regularizer": self.weight.square(),
        }


ARCHITECTURE_CASES = (
    _ArchitectureCase("linreg", LinReg, DefaultLinRegConfig()),
    _ArchitectureCase(
        "nam",
        NAM,
        DefaultNAMConfig(layer_sizes=[8], dropout=0.0),
    ),
    _ArchitectureCase(
        "snam",
        SNAM,
        DefaultSNAMConfig(
            layer_sizes=[8],
            dropout=0.0,
            group_lasso_lambda=0.1,
        ),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "gpnam",
        GPNAM,
        DefaultGPNAMConfig(rff_num_feat=8),
    ),
    _ArchitectureCase(
        "nbm",
        NBM,
        DefaultNBMConfig(
            layer_sizes=[8],
            dropout=0.0,
            bases_dropout=0.0,
            num_bases=4,
            output_penalty=0.1,
        ),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "natt",
        NATT,
        DefaultNATTConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            attn_dropout=0.0,
            transformer_dim_feedforward=16,
            head_layer_sizes=(),
            head_dropout=0.0,
        ),
    ),
    _ArchitectureCase(
        "namformer",
        NAMformer,
        DefaultNAMformerConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            attn_dropout=0.0,
            transformer_dim_feedforward=16,
            head_layer_sizes=(),
            head_dropout=0.0,
        ),
    ),
    _ArchitectureCase(
        "treenam",
        TreeNAM,
        DefaultTreeNAMConfig(tree_depth=2, tree_lamda=0.1),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "ensemble_treenam",
        EnsembleTreeNAM,
        DefaultEnsembleTreeNAMConfig(
            tree_depth=2,
            tree_lamda=0.1,
            num_estimators=2,
        ),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "nodegam",
        NodeGAM,
        DefaultNodeGAMConfig(
            num_trees=4,
            num_layers=1,
            depth=2,
            last_dropout=0.0,
            colsample_bytree=1.0,
            l2_lambda=0.1,
            anneal_steps=10,
            interaction_degree=1,
            quantile_n_quantiles=16,
        ),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "spline_nam",
        SplineNAM,
        DefaultSplineNAMConfig(n_knots=5, smoothing=0.1),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "qnam",
        QNAMBase,
        DefaultQNAMConfig(layer_sizes=[8], dropout=0.0),
        output_dim=3,
        monotone_output=True,
    ),
    _ArchitectureCase(
        "nam_interaction",
        NAM,
        DefaultNAMConfig(
            layer_sizes=[8],
            dropout=0.0,
            interaction_degree=2,
        ),
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "snam_interaction",
        SNAM,
        DefaultSNAMConfig(
            layer_sizes=[8],
            dropout=0.0,
            interaction_degree=2,
            group_lasso_lambda=0.1,
        ),
        expects_penalty=True,
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "nbm_interaction",
        NBM,
        DefaultNBMConfig(
            layer_sizes=[8],
            dropout=0.0,
            bases_dropout=0.0,
            num_bases=4,
            interaction_degree=2,
            output_penalty=0.1,
        ),
        expects_penalty=True,
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "natt_interaction",
        NATT,
        DefaultNATTConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            attn_dropout=0.0,
            transformer_dim_feedforward=16,
            head_layer_sizes=(),
            head_dropout=0.0,
            interaction_degree=2,
        ),
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "namformer_interaction",
        NAMformer,
        DefaultNAMformerConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            attn_dropout=0.0,
            transformer_dim_feedforward=16,
            head_layer_sizes=(),
            head_dropout=0.0,
            interaction_degree=2,
        ),
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "treenam_interaction",
        TreeNAM,
        DefaultTreeNAMConfig(
            tree_depth=2,
            tree_lamda=0.1,
            interaction_degree=2,
        ),
        expects_penalty=True,
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "ensemble_treenam_interaction",
        EnsembleTreeNAM,
        DefaultEnsembleTreeNAMConfig(
            tree_depth=2,
            tree_lamda=0.1,
            num_estimators=2,
            interaction_degree=2,
        ),
        expects_penalty=True,
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "nodegam_interaction",
        NodeGAM,
        DefaultNodeGAMConfig(
            num_trees=4,
            num_layers=1,
            depth=2,
            last_dropout=0.0,
            colsample_bytree=1.0,
            l2_lambda=0.1,
            anneal_steps=10,
            interaction_degree=2,
            quantile_n_quantiles=16,
        ),
        expects_penalty=True,
    ),
    _ArchitectureCase(
        "spline_nam_interaction",
        SplineNAM,
        DefaultSplineNAMConfig(
            n_knots=5,
            smoothing=0.1,
            interaction_degree=2,
        ),
        expects_penalty=True,
        expected_keys=("x:z",),
    ),
    _ArchitectureCase(
        "qnam_interaction",
        QNAMBase,
        DefaultQNAMConfig(
            layer_sizes=[8],
            dropout=0.0,
            interaction_degree=2,
        ),
        output_dim=3,
        monotone_output=True,
        expected_keys=("x:z",),
    ),
)


@pytest.mark.parametrize("case", ARCHITECTURE_CASES, ids=lambda case: case.name)
def test_neural_architecture_forward_backward_contract(case):
    torch.manual_seed(123)
    feature_info = {
        "x": {"dimension": 1},
        "z": {"dimension": 1},
    }
    model = case.model_class(
        cat_feature_info={},
        num_feature_info=feature_info,
        num_classes=case.output_dim,
        config=case.config,
    )
    model.train()
    num_features = {
        "x": torch.linspace(0.05, 0.95, 6).unsqueeze(-1),
        "z": torch.linspace(0.9, 0.1, 6).unsqueeze(-1),
    }

    result = model(num_features=num_features, cat_features={})

    assert {"output", "x", "z"} <= set(result)
    assert set(case.expected_keys) <= set(result)
    assert result["output"].shape == (6, case.output_dim)
    assert torch.isfinite(result["output"]).all()
    for value in result.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()

    penalty_names = [
        name
        for name in result
        if name.endswith("_penalty") or name.endswith("_regularizer")
    ]
    if case.expects_penalty:
        assert penalty_names

    objective = result["output"].square().mean()
    for name in penalty_names:
        objective = objective + result[name]
    objective.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    if case.monotone_output:
        assert torch.all(torch.diff(result["output"], dim=1) >= 0.0)


def test_multi_model_wrapper_aggregates_canonical_penalties():
    wrapper = MultiModelWrapper(
        base_model_class=_PenalizedScalarModel,
        num_models=2,
    )
    result = wrapper(
        num_features={"x": torch.arange(4.0).unsqueeze(-1)},
        cat_features={},
    )

    assert result["output"].shape == (4, 2)
    torch.testing.assert_close(result["output_penalty"], torch.tensor(2.0))

    (result["output"].mean() + result["output_penalty"]).backward()
    assert all(model.weight.grad is not None for model in wrapper.models)


@pytest.mark.parametrize(
    "config_class",
    [DefaultNATTConfig, DefaultNAMformerConfig],
)
def test_transformer_config_activation_modules_are_isolated(config_class):
    first = config_class()
    second = config_class()

    for field_name in (
        "activation",
        "embedding_activation",
        "head_activation",
        "transformer_activation",
    ):
        first_activation = getattr(first, field_name)
        second_activation = getattr(second, field_name)
        assert first_activation is not second_activation
        first_activation.eval()
        assert second_activation.training
