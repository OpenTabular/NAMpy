from __future__ import annotations

import torch
import torch.nn as nn

from nampy.neural.architectures.components.block_masked_additive import (
    BlockMaskedAdditiveNetwork,
)


def test_block_masked_network_matches_independent_and_roundtrips():
    torch.manual_seed(17)
    indices = ((0,), (1, 2), (0, 2, 3))
    block = BlockMaskedAdditiveNetwork(
        input_dim=4,
        term_input_indices=indices,
        hidden_sizes=(5, 3),
        output_dim=2,
        activation=nn.ReLU,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(7, 4)
    expected = block(inputs)

    independent = block.to_independent().eval()
    independent_output = torch.stack(
        [network(inputs[:, term]) for network, term in zip(independent, indices, strict=True)],
        dim=1,
    )
    torch.testing.assert_close(independent_output, expected)

    restored = BlockMaskedAdditiveNetwork(
        input_dim=4,
        term_input_indices=indices,
        hidden_sizes=(5, 3),
        output_dim=2,
        activation=nn.ReLU,
        dropout=0.0,
    ).eval()
    restored.load_independent_(independent)
    torch.testing.assert_close(restored(inputs), expected)

    block.zero_grad(set_to_none=True)
    block(inputs).sum().backward()
    for layer in block.layers:
        assert torch.count_nonzero(layer.weight.grad[~layer.mask]) == 0
