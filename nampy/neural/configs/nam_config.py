# nam_config.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch.nn as nn


@dataclass
class DefaultNAMConfig:
    """
    Configuration class for the default NAM with predefined hyperparameters.

    Parameters
    ----------
    lr : float
        Learning rate for the optimizer.
    lr_patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float
        Factor by which the learning rate will be reduced.
    layer_sizes : list of int
        Sizes of the layers in the MLP.
    feature_layer : str
        Parameterized first-layer type for each feature network: ``"linear"``,
        ``"exu"``, or ``"centered_relu"``.
    activation : type or torch.nn.Module
        Conventional pointwise activation used by linear hidden layers.
    dropout : float
        Dropout rate for regularization.
    norm : str
        Normalization method to be used, if any.
    use_glu : bool
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool
        Whether to use skip connections in the MLP.
    batch_norm : bool
        Whether to use batch normalization in the MLP layers.
    layer_norm : bool
        Whether to use layer normalization in the MLP layers.
    adaptive_width : bool
        Derive each main-effect first-layer width from its transformed training
        cardinality. Disabled by default.
    output_regularization : float
        Mean-squared feature-contribution penalty coefficient.
    l2_regularization : float
        Explicit normalized parameter L2 penalty coefficient, separate from
        optimizer weight decay.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: List[int] = field(default_factory=lambda: [128, 128, 32])
    feature_layer: str = "linear"
    activation: Any = nn.ReLU
    dropout: float = 0.1
    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, ...]]] = None
    intercept: bool = True
    feature_dropout: float = 0.0
    adaptive_width: bool = False
    num_basis_functions: int = 1000
    units_multiplier: int = 2
    feature_widths: Dict[str, int] = field(default_factory=dict)
    feature_output_bias: bool = True
    output_regularization: float = 0.0
    l2_regularization: float = 0.0
    regularize_interactions: bool = False
