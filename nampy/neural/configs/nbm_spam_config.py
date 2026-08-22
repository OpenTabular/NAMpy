"""Configuration for the combined Neural Basis Model and SPAM architecture."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn


@dataclass
class DefaultNBMSPAMConfig:
    """NBM-SPAM defaults with an upstream-compatible NBM basis topology."""

    lr: float = 1e-3
    lr_patience: int = 10
    weight_decay: float = 0.0
    lr_factor: float = 0.1
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
    num_bases: int = 100
    featurizer: str = "conv1d"
    output_penalty: float = 0.0
    ranks: List[int] = field(default_factory=lambda: [100])
    spam_dropout: float = 0.0
    reg_order: int = 2
    lower_order_correction: bool = False
    orthogonal: bool = False
    proximal: bool = False
    regularization_scale: float = 0.0
    basis_l1_regularization: float = 0.0
