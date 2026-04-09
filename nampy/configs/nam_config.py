# nam_config.py
from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn


@dataclass
class DefaultNAMConfig:
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
    layer_sizes : list of int, default=[128, 128, 32]
        Sizes of the layers in the MLP.
    activation : type, default=nn.ReLU
        Activation class for the MLP layers (e.g. nn.ReLU); a new instance is used per layer.
    dropout : float, default=0.1
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
    layer_sizes: List[int] = field(default_factory=lambda: [128, 128, 32])
    activation: type = nn.ReLU
    dropout: float = 0.1
    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    interaction_degree: Optional[int] = None
    intercept: bool = True
    feature_dropout: float = 0.0
