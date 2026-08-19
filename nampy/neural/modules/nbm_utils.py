from typing import List, Optional, Union

import torch
import torch.nn as nn

from .mlp_utils import _make_activation, _make_norm, _resolve_norm_name


class _ResidualBlockNary(nn.Module):
    """
    Feedforward block for N-ary basis networks with optional normalization,
    activation/GLU, dropout, and residual connection.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout: float,
        *,
        activation: Union[type, nn.Module],
        norm_name: Optional[str] = None,
        use_glu: bool = False,
        use_skip: bool = False,
    ) -> None:
        super().__init__()

        modules: list[nn.Module] = [nn.Linear(n_input, n_output)]

        norm_layer = _make_norm(norm_name, n_output)
        if norm_layer is not None:
            modules.append(norm_layer)

        if use_glu:
            modules.append(nn.GLU())
            effective_out = n_output // 2
        else:
            modules.append(_make_activation(activation))
            effective_out = n_output

        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*modules)
        self.use_skip = bool(use_skip and n_input == effective_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            out = out + x
        result: torch.Tensor = out
        return result


class ConceptNNBasesNary(nn.Module):
    """
    Neural network learning basis functions for N-ary interactions.

    Parameters
    ----------
    order : int
        Order of N-ary concept interactions (input dimension per interaction).
    num_bases : int
        Number of basis functions (output dimension).
    layer_sizes : list of int
        Number of units in each hidden layer.
    activation : type or nn.Module
        Activation class (e.g. nn.ReLU) or instance.
    dropout : float
        Dropout rate for hidden layers.
    use_batch_norm : bool
        Whether to use batch normalization.
    use_layer_norm : bool
        Whether to use layer normalization.
    norm : str, optional
        Normalization name (e.g. BatchNorm, LayerNorm, RMSNorm, GroupNorm).
    use_glu : bool
        Whether to use Gated Linear Units; layer_sizes entries must be even.
    skip_connections : bool
        Whether to use skip connections where input and output sizes match.
    """

    def __init__(
        self,
        order: int,
        num_bases: int,
        layer_sizes: List[int],
        activation: Union[type, nn.Module],
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        norm: Optional[str] = None,
        use_glu: bool = False,
        skip_connections: bool = False,
    ) -> None:
        super().__init__()

        if order <= 0:
            raise ValueError("Order of N-ary interactions must be greater than 0.")

        if use_glu:
            for i, s in enumerate(layer_sizes):
                if s % 2 != 0:
                    raise ValueError(
                        f"layer_sizes[{i}]={s} must be even when use_glu=True"
                    )

        norm_name = _resolve_norm_name(
            norm=norm,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        layers = []
        input_dim = order
        for n_hidden in layer_sizes:
            layers.append(
                _ResidualBlockNary(
                    n_input=input_dim,
                    n_output=n_hidden,
                    dropout=dropout,
                    activation=activation,
                    norm_name=norm_name,
                    use_glu=use_glu,
                    use_skip=skip_connections,
                )
            )
            input_dim = n_hidden // 2 if use_glu else n_hidden

        self.hidden_layers = nn.Sequential(*layers)
        self.linear_final = nn.Linear(input_dim, num_bases)

        # Keep final normalization consistent with hidden blocks
        self.norm_final = _make_norm(norm_name, num_bases)
        self.act_final = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.linear_final(x)
        if self.norm_final is not None:
            x = self.norm_final(x)
        x = self.act_final(x)
        return x
