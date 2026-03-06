# mlp_utils.py
from typing import List, Optional

import torch.nn as nn

from .normalization_layers import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)


def _make_norm(norm: Optional[str], size: int) -> Optional[nn.Module]:
    """Instantiate a normalization layer by name, or return None if norm is None."""
    if norm is None:
        return None
    builders = {
        "RMSNorm": lambda: RMSNorm(size),
        "LayerNorm": lambda: LayerNorm(size),
        "BatchNorm": lambda: BatchNorm(size),
        "InstanceNorm": lambda: nn.InstanceNorm1d(size),
        "GroupNorm": lambda: nn.GroupNorm(1, size),
        "LearnableLayerScaling": lambda: LearnableLayerScaling(size),
    }
    if norm not in builders:
        raise ValueError(
            f"Unknown norm {norm!r}. Valid options: {', '.join(builders)}"
        )
    return builders[norm]()


class Linear_skip_block(nn.Module):
    """
    A neural network block that includes a linear layer, an activation function, a dropout layer, and optionally a
    skip connection and batch normalization. The skip connection is added if the input and output feature sizes are equal.

    Parameters
    ----------
    n_input : int
        The number of input features.
    n_output : int
        The number of output features.
    dropout_rate : float
        The rate of dropout to apply for regularization.
    activation_fn : type, optional
        Activation class (e.g. nn.LeakyReLU); a new instance is created per block.
    use_batch_norm : bool, optional
        Whether to apply batch normalization (before activation). Default is False.

    Attributes
    ----------
    fc : torch.nn.Linear
        The linear transformation layer.
    act : torch.nn.Module
        The activation function.
    drop : torch.nn.Dropout
        The dropout layer.
    use_batch_norm : bool
        Indicator of whether batch normalization is used.
    batch_norm : torch.nn.BatchNorm1d, optional
        The batch normalization layer, instantiated if use_batch_norm is True.
    use_skip : bool
        Indicator of whether a skip connection is used.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU,
        use_batch_norm=False,
    ):
        super(Linear_skip_block, self).__init__()

        self.fc = nn.Linear(n_input, n_output)
        self.act = activation_fn() if isinstance(activation_fn, type) else type(activation_fn)()
        self.drop = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm
        self.use_skip = (
            n_input == n_output
        )  # Only use skip connection if input and output sizes are equal

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_output)

    def forward(self, x):
        """
        Defines the forward pass of the Linear_block.

        Parameters
        ----------
        x : Tensor
            The input tensor to the block.

        Returns
        -------
        Tensor
            The output tensor after processing through the linear layer, optional batch norm,
            activation function, and dropout.
        """
        x0 = x
        x = self.fc(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        if self.use_skip:
            x = x + x0
        x = self.drop(x)
        return x


class Linear_block(nn.Module):
    """
    A neural network block that includes a linear layer, an activation function, a dropout layer, and optionally batch normalization.

    Parameters
    ----------
    n_input : int
        The number of input features.
    n_output : int
        The number of output features.
    dropout_rate : float
        The rate of dropout to apply.
    activation_fn : type, optional
        Activation class (e.g. nn.LeakyReLU); a new instance is created per block.
    batch_norm : bool, optional
        Whether to include batch normalization (before activation). Default is False.

    Attributes
    ----------
    block : torch.nn.Sequential
        A sequential container holding the linear layer, activation function, dropout, and optionally batch normalization.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU,
        batch_norm=False,
    ):
        super(Linear_block, self).__init__()

        modules = [nn.Linear(n_input, n_output)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(n_output))
        act = activation_fn() if isinstance(activation_fn, type) else type(activation_fn)()
        modules += [act, nn.Dropout(dropout_rate)]
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        """
        Defines the forward pass of the Linear_block.

        Parameters
        ----------
        x : Tensor
            The input tensor to the block.

        Returns
        -------
        Tensor
            The output tensor after processing through the linear layer, activation function, dropout,
            and optional batch normalization.
        """
        # Pass the input through the block
        return self.block(x)


class _SkipWrapper(nn.Module):
    """Wraps a block with a residual connection (only valid when input/output dims match)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x) + x


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for regression/classification, configurable with
    optional skip connections, batch/layer/norm, and GLU.

    Parameters
    ----------
    n_input_units : int
        The number of units in the input layer.
    hidden_units_list : list of int
        A list specifying the number of units in each hidden layer.
    n_output_units : int
        The number of units in the output layer.
    dropout_rate : float
        The dropout rate used across the MLP.
    use_skip_layers : bool, optional
        Whether to use skip connections in layers where input and output sizes match.
    activation_fn : type, optional
        Activation class (e.g. nn.LeakyReLU); a new instance is used per layer.
    use_batch_norm : bool, optional
        Whether to apply batch normalization (before activation) in each layer.
    use_layer_norm : bool, optional
        Whether to apply nn.LayerNorm after the optional norm in each layer.
    norm : str, optional
        Name of normalization layer: RMSNorm, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, LearnableLayerScaling.
    use_glu : bool, optional
        Whether to use Gated Linear Units; hidden_units_list entries must be even.
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
        if hidden_units_list is None:
            hidden_units_list = [64, 32, 32]
        if use_glu:
            for i, s in enumerate(hidden_units_list):
                if s % 2 != 0:
                    raise ValueError(
                        f"hidden_units_list[{i}]={s} must be even when use_glu=True"
                    )
        super(MLP, self).__init__()
        self.n_input_units = n_input_units
        self.hidden_units_list = hidden_units_list
        self.dropout_rate = dropout_rate
        self.n_output_units = n_output_units

        input_units = n_input_units
        layers = []
        for i, n_hidden in enumerate(hidden_units_list):
            layers.append(
                self._build_block(
                    input_units,
                    n_hidden,
                    dropout_rate,
                    activation_fn=activation_fn,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    norm=norm,
                    use_glu=use_glu,
                    use_skip_layers=use_skip_layers,
                )
            )
            input_units = n_hidden // 2 if use_glu else n_hidden
        self.hidden_layers = nn.Sequential(*layers)
        self.linear_final = nn.Linear(input_units, n_output_units)

    def _build_block(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float,
        *,
        activation_fn,
        use_batch_norm: bool,
        use_layer_norm: bool,
        norm: Optional[str],
        use_glu: bool,
        use_skip_layers: bool,
    ) -> nn.Module:
        """Build a single hidden block: Linear -> [BN] -> [norm] -> [LayerNorm] -> GLU or activation -> Dropout.
        Wrapped in _SkipWrapper when use_skip_layers and n_input == n_output."""
        modules = []
        modules.append(nn.Linear(n_input, n_output))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(n_output))
        norm_layer = _make_norm(norm, n_output)
        if norm_layer is not None:
            modules.append(norm_layer)
        if use_layer_norm:
            modules.append(nn.LayerNorm(n_output))
        if use_glu:
            modules.append(nn.GLU())
        else:
            act = (
                activation_fn()
                if isinstance(activation_fn, type)
                else type(activation_fn)()
            )
            modules.append(act)
        if dropout_rate > 0.0:
            modules.append(nn.Dropout(dropout_rate))
        seq = nn.Sequential(*modules)
        effective_out = n_output // 2 if use_glu else n_output
        if use_skip_layers and n_input == effective_out:
            return _SkipWrapper(seq)
        return seq

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Parameters
        ----------
        x : Tensor
            The input tensor to the MLP.

        Returns
        -------
        Tensor
            The output predictions of the model for regression tasks.
        """
        x = self.hidden_layers(x)
        x = self.linear_final(x)
        return x
