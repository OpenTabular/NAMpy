from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch.nn as nn


NaryExplicit = Dict[str, List[Tuple[int, ...]]]
NarySpec = Optional[Union[NaryExplicit, List[int], Tuple[int, ...]]]


@dataclass
class DefaultNBMConfig:
    """
    Configuration class for the default NBM with predefined hyperparameters.

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
        Sizes of the layers in the basis MLP.
    activation_fn : type, default=nn.ReLU
        Activation class for the basis MLP layers (e.g. nn.ReLU); a new instance is used per layer.
    dropout_rate : float, default=0.1
        Dropout rate for regularization.
    bases_dropout : float, default=0.1
        Dropout rate for entire basis function outputs.
    norm : str, default=None
        Normalization method (e.g. BatchNorm, LayerNorm, RMSNorm, GroupNorm).
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the basis MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the basis MLP.
    num_subnets : int, default=1
        Number of sub-networks to learn basis functions.
    batch_norm : bool, default=False
        Whether to use batch normalization in the basis MLP layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the basis MLP layers.
    intercept : bool, default=True
        Whether to use a learnable intercept parameter.
    feature_dropout : float, default=0.0
        Probability for feature-level dropout (drops whole feature outputs).
    interaction_degree : int, optional
        Degree of feature interactions to model; if None, no interactions are added.
    num_bases : int, default=100
        Number of shared basis functions.
    nary : dict, optional
        N-ary interaction index sets; if None, unary (order 1) is used.
    order : int, default=1
        Order of N-ary concept interactions when nary is not provided.
    output_penalty : float, default=0.0
        Coefficient for L2 penalty on term scores (added to task loss when > 0).
    """
    
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: List[int] = field(default_factory=lambda: [128, 128, 32])
    activation_fn: type = nn.ReLU
    dropout_rate: float = 0.1
    bases_dropout: float = 0.1
    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    num_subnets: int = 1
    batch_norm: bool = False
    layer_norm: bool = False
    intercept: bool = True
    feature_dropout: float = 0.0
    interaction_degree: Optional[int] = None
    num_bases: int = 100
    nary: NarySpec = None
    order: int = 1
    output_penalty: float = 0.0