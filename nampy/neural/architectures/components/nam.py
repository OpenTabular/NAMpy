"""Feature-network layers used by neural additive models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP


class ExU(nn.Module):
    """Exponentially transformed unit from the Neural Additive Models paper.

    For a scalar feature this is exactly ``clip(exp(beta) * (x - center), 0, 1)``.
    The matrix form below extends the same operation to grouped feature inputs.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.beta = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.center = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.beta, mean=4.0, std=0.5, a=3.0, b=5.0)
        nn.init.trunc_normal_(self.center, mean=0.0, std=0.5, a=-1.0, b=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Google's scalar-feature layer learns one center per hidden unit.
        # Keeping an input axis here preserves that exact parameterization for
        # ``in_features == 1`` and extends it naturally to grouped inputs.
        outputs = torch.sum(
            (inputs.unsqueeze(-1) - self.center) * torch.exp(self.beta), dim=1
        )
        return torch.clamp(outputs, min=0.0, max=1.0)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class CenteredReLU(nn.Module):
    """Learnable centered linear layer followed by ReLU."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.center = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.trunc_normal_(self.center, mean=0.0, std=0.5, a=-1.0, b=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.sum(
            (inputs.unsqueeze(-1) - self.center) * self.weight, dim=1
        )
        return torch.relu(outputs)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class NAMFeatureNN(nn.Module):
    """A feature network with an ExU or centered-ReLU first layer."""

    def __init__(
        self,
        n_input_units: int,
        hidden_units_list: list[int],
        n_output_units: int,
        *,
        feature_layer: str,
        dropout: float,
        output_bias: bool,
        use_skip_layers: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        norm: str | None = None,
        use_glu: bool = False,
    ):
        super().__init__()
        if not hidden_units_list:
            raise ValueError("ExU and centered-ReLU feature networks need a hidden layer.")
        first_width = int(hidden_units_list[0])
        builders = {
            "exu": ExU,
            "centered_relu": CenteredReLU,
        }
        if feature_layer not in builders:
            raise ValueError(
                "feature_layer must be 'exu' or 'centered_relu'; "
                f"got {feature_layer!r}."
            )
        self.first_layer = builders[feature_layer](n_input_units, first_width)
        self.first_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        remaining = list(hidden_units_list[1:])
        if remaining:
            self.tail = MLP(
                n_input_units=first_width,
                hidden_units_list=remaining,
                n_output_units=n_output_units,
                dropout=dropout,
                use_skip_layers=use_skip_layers,
                activation=nn.ReLU,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                norm=norm,
                use_glu=use_glu,
                output_bias=output_bias,
            )
        else:
            self.tail = nn.Linear(first_width, n_output_units, bias=output_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tail(self.first_dropout(self.first_layer(inputs)))
