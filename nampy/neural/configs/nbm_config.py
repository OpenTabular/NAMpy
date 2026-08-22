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
    lr : float
        Learning rate for the optimizer.
    lr_patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float
        Factor by which the learning rate will be reduced.
    layer_sizes : list of int
        Sizes of the layers in the basis MLP.
    activation : type
        Activation class for the basis MLP layers (e.g. nn.ReLU); a new instance is used per layer.
    dropout : float
        Dropout rate for regularization.
    bases_dropout : float
        Dropout rate for entire basis function outputs.
    norm : str
        Normalization method (e.g. BatchNorm, LayerNorm, RMSNorm, GroupNorm).
    use_glu : bool
        Whether to use Gated Linear Units (GLU) in the basis MLP.
    skip_connections : bool
        Whether to use skip connections in the basis MLP.
    num_subnets : int
        Number of sub-networks to learn basis functions.
    batch_norm : bool
        Whether to use batch normalization in the basis MLP layers.
    layer_norm : bool
        Whether to use layer normalization in the basis MLP layers.
    intercept : bool
        Whether to use a learnable intercept parameter.
    feature_dropout : float
        Probability for feature-level dropout (drops whole feature outputs).
    interaction_degree : int, optional
        Degree of feature interactions to model; if None, no interactions are added.
    num_bases : int
        Number of shared basis functions.
    nary : dict, optional
        N-ary interaction index sets; if None, unary (order 1) is used.
    order : int
        Order of N-ary concept interactions when nary is not provided.
    output_penalty : float
        Coefficient for L2 penalty on term scores (added to task loss when > 0).
    featurizer : {"conv1d", "einsum"}
        Per-term combination of shared basis responses. Grouped ``conv1d``
        matches the released dense implementation.
    sparse : bool
        Use the released active-tuple sparse execution topology.
    nary_ignore_input : float or dict
        Sentinel ignored by sparse execution, globally or per interaction order.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    # Defaults mirror ConceptNBMNary in the released NBM-SPAM implementation.
    layer_sizes: List[int] = field(default_factory=lambda: [256, 128, 128])
    activation: type = nn.ReLU
    dropout: float = 0.0
    bases_dropout: float = 0.0
    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    num_subnets: int = 1
    batch_norm: bool = True
    layer_norm: bool = False
    intercept: bool = True
    feature_dropout: float = 0.0
    interaction_degree: Optional[int] = None
    num_bases: int = 100
    nary: NarySpec = None
    order: int = 1
    output_penalty: float = 0.0
    featurizer: str = "conv1d"
    sparse: bool = False
    nary_ignore_input: Union[float, Dict[str, float]] = 0.0
