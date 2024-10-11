from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultNodeGAMConfig:
    """
    Configuration class for the default NAM with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the MLP.
    activation : callable, default=nn.SELU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to skip layers in the MLP.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the MLP.
    batch_norm : bool, default=False
        Whether to use batch normalization in the MLP layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the MLP layers.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    lr_decay_steps: int = 300
    lr_warmup_steps: int = 100
    max_steps: int = 10000
    num_trees: int = 200
    num_layers: int = 2
    depth: int = 3
    tree_dim: int = 1
    additional_tree_output_dim: int = 0
    last_dropout: float = 0.3
    colsample_bytree: float = 0.7
    l2_lambda: float = 0.0
    selectors_detach: bool = False
    fs_normalize: bool = False
    ga2m: bool = False