import copy
from typing import List, Optional

import torch.nn as nn

from .normalization_layers import (
    BatchNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)


def _make_activation(activation_fn) -> nn.Module:
    """Instantiate an activation from a class or deep-copy an instance."""
    return (
        activation_fn()
        if isinstance(activation_fn, type)
        else copy.deepcopy(activation_fn)
    )


def _make_norm(norm: Optional[str], size: int) -> Optional[nn.Module]:
    """Instantiate a normalization layer by name, or return None if norm is None."""
    if norm is None:
        return None

    builders = {
        "RMSNorm": lambda: RMSNorm(size),
        "LayerNorm": lambda: LayerNorm(size),
        "BatchNorm": lambda: BatchNorm(size),
        # These two use native PyTorch modules for tabular [N, D] inputs.
        "InstanceNorm": lambda: nn.InstanceNorm1d(size),
        "GroupNorm": lambda: nn.GroupNorm(1, size),
        "LearnableLayerScaling": lambda: LearnableLayerScaling(size),
    }

    if norm not in builders:
        raise ValueError(
            f"Unknown norm {norm!r}. Valid options: {', '.join(builders)}"
        )
    return builders[norm]()


def _resolve_norm_name(
    norm: Optional[str],
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
) -> Optional[str]:
    """
    Resolve the effective normalization choice.

    Rules
    -----
    - If `norm` is provided, it takes precedence and the boolean flags must be False.
    - Otherwise, `use_batch_norm=True` maps to "BatchNorm".
    - Otherwise, `use_layer_norm=True` maps to "LayerNorm".
    - At most one normalization mode may be active.
    """
    if norm is not None:
        if use_batch_norm or use_layer_norm:
            raise ValueError(
                "Use either `norm` or one of (`use_batch_norm`, `use_layer_norm`), not both."
            )
        return norm

    if use_batch_norm and use_layer_norm:
        raise ValueError(
            "Only one of `use_batch_norm` and `use_layer_norm` may be True."
        )

    if use_batch_norm:
        return "BatchNorm"
    if use_layer_norm:
        return "LayerNorm"
    return None


class _ResidualBlock(nn.Module):
    """
    Generic feedforward block with optional normalization, activation/GLU,
    dropout, and residual connection.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float,
        *,
        activation_fn=nn.LeakyReLU,
        norm_name: Optional[str] = None,
        use_glu: bool = False,
        use_skip: bool = False,
    ):
        super().__init__()

        modules = [nn.Linear(n_input, n_output)]

        norm_layer = _make_norm(norm_name, n_output)
        if norm_layer is not None:
            modules.append(norm_layer)

        if use_glu:
            modules.append(nn.GLU())
            effective_out = n_output // 2
        else:
            modules.append(_make_activation(activation_fn))
            effective_out = n_output

        if dropout_rate > 0.0:
            modules.append(nn.Dropout(dropout_rate))

        self.block = nn.Sequential(*modules)
        self.use_skip = bool(use_skip and n_input == effective_out)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class Linear_skip_block(nn.Module):
    """
    Backward-compatible wrapper for a linear block with optional batch norm and
    residual connection when input/output sizes match.

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_output : int
        Number of output features.
    dropout_rate : float
        Dropout rate.
    activation_fn : type or nn.Module, optional
        Activation class or instance.
    use_batch_norm : bool, optional
        Whether to apply batch normalization.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU,
        use_batch_norm=False,
    ):
        super().__init__()
        norm_name = _resolve_norm_name(
            norm=None, use_batch_norm=use_batch_norm, use_layer_norm=False
        )
        self.block = _ResidualBlock(
            n_input=n_input,
            n_output=n_output,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            norm_name=norm_name,
            use_glu=False,
            use_skip=True,
        )

    def forward(self, x):
        return self.block(x)


class Linear_block(nn.Module):
    """
    Backward-compatible wrapper for a linear block with optional batch norm.

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_output : int
        Number of output features.
    dropout_rate : float
        Dropout rate.
    activation_fn : type or nn.Module, optional
        Activation class or instance.
    batch_norm : bool, optional
        Whether to apply batch normalization.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU,
        batch_norm=False,
    ):
        super().__init__()
        norm_name = _resolve_norm_name(
            norm=None, use_batch_norm=batch_norm, use_layer_norm=False
        )
        self.block = _ResidualBlock(
            n_input=n_input,
            n_output=n_output,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            norm_name=norm_name,
            use_glu=False,
            use_skip=False,
        )

    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron for regression/classification with configurable
    hidden layers, normalization, skip connections, and GLU.

    Parameters
    ----------
    n_input_units : int
        Number of input units.
    hidden_units_list : list of int, optional
        Hidden layer sizes.
    n_output_units : int, default=1
        Number of output units.
    dropout_rate : float, default=0.1
        Dropout rate.
    use_skip_layers : bool, default=False
        Whether to use residual connections when dimensions match.
    activation_fn : type or nn.Module, default=nn.LeakyReLU
        Activation class or instance.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
    use_layer_norm : bool, default=False
        Whether to use layer normalization.
    norm : str, optional
        Explicit normalization name. Valid options are:
        RMSNorm, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
        LearnableLayerScaling.
    use_glu : bool, default=False
        Whether to use GLU in hidden layers. Hidden sizes must then be even.
    """

    def __init__(
        self,
        n_input_units: int,
        hidden_units_list: Optional[List[int]] = None,
        n_output_units: int = 1,
        dropout_rate: float = 0.1,
        use_skip_layers: bool = False,
        activation_fn=nn.LeakyReLU,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        norm: Optional[str] = None,
        use_glu: bool = False,
    ):
        super().__init__()

        if hidden_units_list is None:
            hidden_units_list = [64, 32, 32]

        if use_glu:
            for i, s in enumerate(hidden_units_list):
                if s % 2 != 0:
                    raise ValueError(
                        f"hidden_units_list[{i}]={s} must be even when use_glu=True"
                    )

        norm_name = _resolve_norm_name(
            norm=norm,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.n_input_units = n_input_units
        self.hidden_units_list = list(hidden_units_list)
        self.dropout_rate = dropout_rate
        self.n_output_units = n_output_units
        self.use_skip_layers = use_skip_layers
        self.use_glu = use_glu
        self.norm_name = norm_name

        layers = []
        input_units = n_input_units

        for n_hidden in self.hidden_units_list:
            layers.append(
                _ResidualBlock(
                    n_input=input_units,
                    n_output=n_hidden,
                    dropout_rate=dropout_rate,
                    activation_fn=activation_fn,
                    norm_name=norm_name,
                    use_glu=use_glu,
                    use_skip=use_skip_layers,
                )
            )
            input_units = n_hidden // 2 if use_glu else n_hidden

        self.hidden_layers = nn.Sequential(*layers)
        self.linear_final = nn.Linear(input_units, n_output_units)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.linear_final(x)
        return x