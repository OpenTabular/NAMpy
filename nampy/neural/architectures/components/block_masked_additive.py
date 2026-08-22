"""Parallel block-masked execution for equal-depth additive term networks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP, make_activation


class MaskedLinear(nn.Module):
    """Dense linear storage with a structural mask applied in every forward pass."""

    def __init__(self, n_input: int, n_output: int, mask: torch.Tensor):
        super().__init__()
        if mask.shape != (n_output, n_input):
            raise ValueError("MaskedLinear mask shape must match [output, input].")
        self.weight = nn.Parameter(torch.empty(n_output, n_input))
        self.bias = nn.Parameter(torch.empty(n_output))
        self.register_buffer("mask", mask.to(dtype=torch.bool))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / n_input**0.5
        nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight * self.mask, self.bias)

    def active_l1(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.weight * self.mask)) + torch.sum(
            torch.abs(self.bias)
        )

    def active_parameter_count(self) -> int:
        bias_count = self.bias.numel() if self.bias.requires_grad else 0
        return int(torch.count_nonzero(self.mask)) + int(bias_count)


class BlockMaskedAdditiveNetwork(nn.Module):
    """Evaluate many plain MLP term functions in one block-masked stack.

    Every term has the same hidden widths and output width, while its first
    layer reads only the transformed columns assigned to that term. Later
    layers are block diagonal. This is the stable form of SIAN's block-sparse
    representation; the matrices remain dense but all inactive entries are
    structurally masked during forward and backward computation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        term_input_indices: Sequence[Sequence[int]],
        hidden_sizes: Sequence[int],
        output_dim: int = 1,
        activation=nn.ReLU,
        dropout: float = 0.0,
        output_bias: bool = True,
        scale_later_layers: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.term_input_indices = tuple(
            tuple(int(index) for index in indices) for indices in term_input_indices
        )
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.output_bias = bool(output_bias)
        self.scale_later_layers = bool(scale_later_layers)
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("Block-masked input and output dimensions must be positive.")
        if not self.term_input_indices:
            raise ValueError("At least one additive term is required.")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        for indices in self.term_input_indices:
            if not indices or min(indices) < 0 or max(indices) >= self.input_dim:
                raise ValueError("Each term must reference valid transformed columns.")

        self.n_terms = len(self.term_input_indices)
        widths = [*self.hidden_sizes, self.output_dim]
        layers = []
        previous_width = self.input_dim
        for layer_index, term_width in enumerate(widths):
            combined_width = self.n_terms * term_width
            mask = torch.zeros(combined_width, previous_width, dtype=torch.bool)
            for term_index, indices in enumerate(self.term_input_indices):
                row_start = term_index * term_width
                row_slice = slice(row_start, row_start + term_width)
                if layer_index == 0:
                    mask[row_slice, list(indices)] = True
                else:
                    previous_term_width = widths[layer_index - 1]
                    column_start = term_index * previous_term_width
                    column_slice = slice(
                        column_start, column_start + previous_term_width
                    )
                    mask[row_slice, column_slice] = True
            layer = MaskedLinear(previous_width, combined_width, mask)
            if self.scale_later_layers and layer_index > 0:
                with torch.no_grad():
                    scale = self.n_terms**0.5
                    layer.weight.mul_(scale)
                    layer.bias.mul_(scale)
            if layer_index == len(widths) - 1 and not self.output_bias:
                layer.bias.requires_grad_(False)
                layer.bias.zero_()
            layers.append(layer)
            previous_width = combined_width
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(
            make_activation(activation) for _ in self.hidden_sizes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < len(self.activations):
                hidden = self.activations[index](hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden.reshape(inputs.shape[0], self.n_terms, self.output_dim)

    def active_l1(self) -> torch.Tensor:
        return sum(
            (layer.active_l1() for layer in self.layers),
            start=self.layers[0].weight.new_zeros(()),
        )

    def active_parameter_count(self) -> int:
        return sum(layer.active_parameter_count() for layer in self.layers)

    def to_independent(self) -> nn.ModuleList:
        """Return independent MLPs with exactly the same active parameters."""
        networks = nn.ModuleList(
            MLP(
                n_input_units=len(indices),
                hidden_units_list=list(self.hidden_sizes),
                n_output_units=self.output_dim,
                dropout=self.dropout,
                activation=self.activations[0],
                output_bias=self.output_bias,
            ).to(self.layers[0].weight.device)
            for indices in self.term_input_indices
        )
        for term_index, (indices, network) in enumerate(
            zip(self.term_input_indices, networks, strict=True)
        ):
            for layer_index, source in enumerate(self.layers):
                term_width = (
                    self.hidden_sizes[layer_index]
                    if layer_index < len(self.hidden_sizes)
                    else self.output_dim
                )
                row_start = term_index * term_width
                rows = slice(row_start, row_start + term_width)
                target = (
                    network.hidden_layers[layer_index].block[0]
                    if layer_index < len(self.hidden_sizes)
                    else network.linear_final
                )
                with torch.no_grad():
                    if layer_index == 0:
                        target.weight.copy_(source.weight[rows][:, list(indices)])
                    else:
                        previous_width = self.hidden_sizes[layer_index - 1]
                        column_start = term_index * previous_width
                        columns = slice(column_start, column_start + previous_width)
                        target.weight.copy_(source.weight[rows, columns])
                    if target.bias is not None:
                        target.bias.copy_(source.bias[rows])
        return networks

    def load_independent_(self, networks: Sequence[MLP]) -> None:
        """Load compatible independent MLPs into this block representation."""
        if len(networks) != self.n_terms:
            raise ValueError("Independent network count does not match term count.")
        with torch.no_grad():
            for source in self.layers:
                source.weight.zero_()
            for term_index, (indices, network) in enumerate(
                zip(self.term_input_indices, networks, strict=True)
            ):
                if tuple(network.hidden_units_list) != self.hidden_sizes:
                    raise ValueError("Independent network hidden widths do not match.")
                if network.n_output_units != self.output_dim:
                    raise ValueError("Independent network output width does not match.")
                for layer_index, target in enumerate(self.layers):
                    term_width = (
                        self.hidden_sizes[layer_index]
                        if layer_index < len(self.hidden_sizes)
                        else self.output_dim
                    )
                    row_start = term_index * term_width
                    rows = slice(row_start, row_start + term_width)
                    source = (
                        network.hidden_layers[layer_index].block[0]
                        if layer_index < len(self.hidden_sizes)
                        else network.linear_final
                    )
                    if layer_index == 0:
                        for local_index, global_index in enumerate(indices):
                            target.weight[rows, global_index].copy_(
                                source.weight[:, local_index]
                            )
                    else:
                        previous_width = self.hidden_sizes[layer_index - 1]
                        column_start = term_index * previous_width
                        columns = slice(column_start, column_start + previous_width)
                        target.weight[rows, columns].copy_(source.weight)
                    if source.bias is not None:
                        target.bias[rows].copy_(source.bias)
                    else:
                        target.bias[rows].zero_()


__all__ = ["BlockMaskedAdditiveNetwork", "MaskedLinear"]
