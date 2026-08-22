"""Small reference networks used before model-specific interaction selection."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ReferenceMLP(nn.Module):
    """Plain ReLU MLP matching the reference model used by SIAN."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int):
        super().__init__()
        sizes = [int(input_dim), *(int(size) for size in hidden_sizes), int(output_dim)]
        if any(size < 1 for size in sizes):
            raise ValueError("Reference MLP dimensions must be positive.")
        layers = []
        for index, (n_input, n_output) in enumerate(
            zip(sizes, sizes[1:], strict=False)
        ):
            layers.append(nn.Linear(n_input, n_output))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def fit_reference_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    objective_kind: str,
    hidden_sizes: Sequence[int] = (256, 128, 64),
    output_index: int = 0,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 5e-3,
    weight_decay: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    random_state: int = 0,
    device: str | torch.device = "cpu",
) -> ReferenceMLP:
    """Fit a detector reference model without coupling it to final objectives."""
    if inputs.ndim != 2:
        raise ValueError("Reference inputs must have shape [rows, columns].")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Reference epochs and batch_size must be positive.")
    kind = str(objective_kind).lower()
    y = targets
    if kind in {"regression", "distributional"}:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if output_index >= y.shape[1]:
            raise ValueError(
                f"selection output_index={output_index} exceeds target width {y.shape[1]}."
            )
        y = y[:, output_index : output_index + 1].float()
        output_dim = 1
    elif kind == "binary":
        y = y.reshape(-1, 1).float()
        output_dim = 1
    elif kind == "multiclass":
        y = y.reshape(-1).long()
        output_dim = int(torch.max(y).item()) + 1
    else:
        raise ValueError(f"Unsupported reference objective kind {objective_kind!r}.")

    tensors = [inputs.float(), y]
    if sample_weight is None:
        sample_weight = torch.ones(inputs.shape[0])
    tensors.append(sample_weight.reshape(-1).float())
    if offset is None:
        offset = torch.zeros(inputs.shape[0], 1)
    tensors.append(offset.reshape(inputs.shape[0], -1).float())

    generator = torch.Generator().manual_seed(int(random_state))
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    torch.manual_seed(int(random_state))
    model = ReferenceMLP(inputs.shape[1], hidden_sizes, output_dim).to(device)
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    model.train()
    for _ in range(int(epochs)):
        for batch_inputs, batch_targets, weights, batch_offset in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            weights = weights.to(device)
            batch_offset = batch_offset.to(device)
            predictions = model(batch_inputs)
            if kind == "multiclass":
                losses = F.cross_entropy(predictions, batch_targets, reduction="none")
            elif kind == "binary":
                losses = F.binary_cross_entropy_with_logits(
                    predictions + batch_offset[:, :1],
                    batch_targets,
                    reduction="none",
                ).reshape(-1)
            else:
                losses = F.mse_loss(
                    predictions + batch_offset[:, :1],
                    batch_targets,
                    reduction="none",
                ).reshape(-1)
            normalizer = torch.clamp(weights.sum(), min=torch.finfo(weights.dtype).eps)
            loss = torch.sum(losses * weights) / normalizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


__all__ = ["ReferenceMLP", "fit_reference_model"]
