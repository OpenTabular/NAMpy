"""Sparse Interaction Additive Network architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from ..configs.sian_config import DefaultSIANConfig
from .components.base_model import BaseModel
from .components.block_masked_additive import BlockMaskedAdditiveNetwork
from .components.interactions import resolve_interactions
from .components.mlp import MLP


class MuPResidualMLP(nn.Module):
    """SIAN's optional maximal-update-style unrestricted residual network."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int):
        super().__init__()
        sizes = [int(input_dim), *(int(size) for size in hidden_sizes), int(output_dim)]
        if len(sizes) < 3 or any(size < 1 for size in sizes):
            raise ValueError("MuP residual networks require positive hidden widths.")
        self.layers = nn.ModuleList(
            nn.Linear(n_input, n_output)
            for n_input, n_output in zip(sizes, sizes[1:], strict=False)
        )
        for index, layer in enumerate(self.layers):
            if index == 0:
                # Preserve the released MuP_Relu_DNN initialization order. Its
                # fan-out draw is immediately replaced by the fan-in draw.
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_out", nonlinearity="relu"
                )
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(
                layer.weight, a=0, mode="fan_in", nonlinearity="relu"
            )
        self.first_width = sizes[1]
        self.final_input_width = sizes[-2]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index == 0:
                hidden = hidden * math.sqrt(self.first_width)
            if index < len(self.layers) - 1:
                hidden = torch.relu(hidden)
            else:
                hidden = hidden / math.sqrt(self.final_input_width)
        return hidden


class SIAN(BaseModel):
    """Higher-order additive ReLU network with an optional block-masked backend."""

    estimator_fitted_attributes = ("term_names_", "execution_mode_")
    extra_reserved_feature_names = ("residual",)

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultSIANConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultSIANConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = int(num_classes)
        self.feature_names = tuple(num_feature_info) + tuple(cat_feature_info)
        if not self.feature_names:
            raise ValueError("SIAN requires at least one feature.")

        self.layer_sizes = tuple(
            int(size) for size in self.hparams.get("layer_sizes", config.layer_sizes)
        )
        self.activation = self.hparams.get("activation", config.activation)
        self.dropout = float(self.hparams.get("dropout", config.dropout))
        self.feature_output_bias = bool(
            self.hparams.get("feature_output_bias", config.feature_output_bias)
        )
        self.l1_regularization = float(
            self.hparams.get("l1_regularization", config.l1_regularization)
        )
        if self.l1_regularization < 0:
            raise ValueError("l1_regularization must be non-negative.")

        interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        interactions = self.hparams.get("interactions", config.interactions)
        resolved_interactions = resolve_interactions(
            self.feature_names, interaction_degree, interactions
        )
        self.terms = tuple((name,) for name in self.feature_names) + tuple(
            resolved_interactions
        )
        self.term_names_ = tuple(":".join(term) for term in self.terms)

        self._feature_columns, input_dim = self._resolve_feature_columns(
            num_feature_info, cat_feature_info
        )
        self._term_input_indices = tuple(
            tuple(
                column
                for feature_name in term
                for column in self._feature_columns[feature_name]
            )
            for term in self.terms
        )
        self.input_dim = input_dim

        self.intercept: nn.Parameter | None
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.intercept = None

        requested_mode = str(
            self.hparams.get("execution_mode", config.execution_mode)
        ).lower()
        if requested_mode not in {"block_masked", "independent"}:
            raise ValueError("execution_mode must be 'block_masked' or 'independent'.")
        self.execution_mode_ = requested_mode
        self.block_network: BlockMaskedAdditiveNetwork | None = None
        self.term_networks: nn.ModuleDict | None = None
        if requested_mode == "block_masked":
            self.block_network = self._make_block_network()
        else:
            self.term_networks = self._make_independent_networks()

        self.residual_network_enabled = bool(
            self.hparams.get("residual_network", config.residual_network)
        )
        self.residual_network: nn.Module | None = None
        if self.residual_network_enabled:
            residual_sizes = self.hparams.get(
                "residual_layer_sizes", config.residual_layer_sizes
            )
            self.residual_network = MuPResidualMLP(
                self.input_dim,
                residual_sizes,
                self.num_classes,
            )

    @staticmethod
    def _resolve_feature_columns(num_info, cat_info):
        columns = {}
        start = 0
        for name, info in [*num_info.items(), *cat_info.items()]:
            width = int(info["dimension"])
            if width < 1:
                raise ValueError(f"Feature {name!r} must have positive dimension.")
            columns[name] = tuple(range(start, start + width))
            start += width
        return columns, start

    def _make_block_network(self) -> BlockMaskedAdditiveNetwork:
        return BlockMaskedAdditiveNetwork(
            input_dim=self.input_dim,
            term_input_indices=self._term_input_indices,
            hidden_sizes=self.layer_sizes,
            output_dim=self.num_classes,
            activation=self.activation,
            dropout=self.dropout,
            output_bias=self.feature_output_bias,
            scale_later_layers=True,
        )

    def _make_independent_networks(
        self, networks: Sequence[MLP] | None = None
    ) -> nn.ModuleDict:
        if networks is None:
            networks = self._make_block_network().to_independent()
        return nn.ModuleDict(dict(zip(self.term_names_, networks, strict=True)))

    def _concatenate(self, num_features, cat_features) -> torch.Tensor:
        all_features = {**num_features, **cat_features}
        return torch.cat(
            [all_features[name].float() for name in self.feature_names], dim=1
        )

    def _term_outputs(
        self,
        inputs: torch.Tensor,
        num_features: Mapping[str, torch.Tensor],
        cat_features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.execution_mode_ == "block_masked":
            assert self.block_network is not None
            return self.block_network(inputs)
        assert self.term_networks is not None
        all_features = {**num_features, **cat_features}
        outputs = []
        for term, name in zip(self.terms, self.term_names_, strict=True):
            term_inputs = torch.cat(
                [all_features[feature].float() for feature in term], dim=1
            )
            outputs.append(self.term_networks[name](term_inputs))
        return torch.stack(outputs, dim=1)

    def _active_l1(self) -> torch.Tensor:
        if self.execution_mode_ == "block_masked":
            assert self.block_network is not None
            penalty = self.block_network.active_l1()
        else:
            assert self.term_networks is not None
            penalty = sum(
                (
                    parameter.abs().sum()
                    for network in self.term_networks.values()
                    for parameter in network.parameters()
                ),
                start=next(self.parameters()).new_zeros(()),
            )
        if self.residual_network is not None:
            penalty = penalty + sum(
                (parameter.abs().sum() for parameter in self.residual_network.parameters()),
                start=penalty.new_zeros(()),
            )
        return penalty

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        inputs = self._concatenate(num_features, cat_features)
        term_outputs = self._term_outputs(inputs, num_features, cat_features)
        output = term_outputs.sum(dim=1)
        result = {
            name: term_outputs[:, index, :]
            for index, name in enumerate(self.term_names_)
        }
        if self.residual_network is not None:
            residual = self.residual_network(inputs)
            result["residual"] = residual
            output = output + residual
        if self.intercept is not None:
            output = output + self.intercept
            result["intercept"] = self.intercept
        result["output"] = output
        if self.l1_regularization > 0:
            result["parameter_regularizer"] = (
                self.l1_regularization * self._active_l1()
            )
        return result

    def compress(self) -> None:
        """Convert block-masked term weights to independent subnetworks."""
        if self.execution_mode_ == "independent":
            return
        assert self.block_network is not None
        self.term_networks = self._make_independent_networks(
            self.block_network.to_independent()
        )
        self.block_network = None
        self.execution_mode_ = "independent"

    def block_mask(self) -> None:
        """Convert independent subnetworks to block-masked execution."""
        if self.execution_mode_ == "block_masked":
            return
        assert self.term_networks is not None
        block_network = self._make_block_network().to(next(self.parameters()).device)
        block_network.load_independent_(list(self.term_networks.values()))
        self.block_network = block_network
        self.term_networks = None
        self.execution_mode_ = "block_masked"

    def complexity_metadata(self) -> dict[str, int]:
        """Report term structure and active parameters separately from storage."""
        if self.execution_mode_ == "block_masked":
            assert self.block_network is not None
            active = self.block_network.active_parameter_count()
        else:
            assert self.term_networks is not None
            active = sum(
                parameter.numel()
                for network in self.term_networks.values()
                for parameter in network.parameters()
                if parameter.requires_grad
            )
        if self.intercept is not None:
            active += self.intercept.numel()
        if self.residual_network is not None:
            active += sum(
                parameter.numel()
                for parameter in self.residual_network.parameters()
                if parameter.requires_grad
            )
        return {
            "active_parameters": int(active),
            "additive_terms": len(self.terms),
            "interaction_terms": sum(len(term) > 1 for term in self.terms),
            "max_interaction_order": max(map(len, self.terms)),
        }


__all__ = ["MuPResidualMLP", "SIAN"]
