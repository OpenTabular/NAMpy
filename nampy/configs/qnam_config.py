from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn


@dataclass
class DefaultQNAMConfig:
    """
    Configuration for Quantile Neural Additive Models (QNAM).

    Parameters
    ----------
    lr : float, default=1e-4
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no validation improvement before reducing the learning rate.
    weight_decay : float, default=1e-6
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Multiplicative factor used by the learning-rate scheduler.

    layer_sizes : list of int, default=[128, 128, 32]
        Hidden-layer widths for each feature subnet.

    activation_fn : type, default=nn.ReLU
        Activation class used in the hidden layers.

    dropout_rate : float, default=0.1
        Dropout rate inside each feature subnet.

    norm : str or None, default=None
        Optional normalization layer name.
        Valid values depend on `mlp_utils._make_norm`, e.g.
        "BatchNorm", "LayerNorm", "RMSNorm", "LearnableLayerScaling".

    use_glu : bool, default=False
        Whether to use GLU in hidden layers. If True, all hidden sizes must be even.

    skip_connections : bool, default=False
        Whether to use residual connections when layer dimensions match.

    batch_norm : bool, default=False
        Convenience flag for batch normalization. Mutually exclusive with `norm`
        and `layer_norm` in the cleaned-up MLP utilities.

    layer_norm : bool, default=False
        Convenience flag for layer normalization. Mutually exclusive with `norm`
        and `batch_norm` in the cleaned-up MLP utilities.

    intercept : bool, default=True
        Whether to include a learnable monotone intercept across quantiles.

    feature_dropout : float, default=0.0
        Dropout probability applied at the whole-feature contribution level.

    interaction_degree : int or None, default=None
        Maximum interaction degree. If None or < 2, only main effects are used.
        If 2, pairwise interactions are included, etc.

    monotone_transform : str, default="softplus"
        Transformation function to enforce monotonicity. Options: "softplus", "exponential".

    min_increment : float, default=0.01
        Minimum increment for the monotone transform. Only used if `monotone_transform` is "softplus".
    """

    lr: float = 1e-4
    lr_patience: int = 10
    weight_decay: float = 1e-6
    lr_factor: float = 0.1

    layer_sizes: List[int] = field(default_factory=lambda: [128, 128, 32])
    activation_fn: type = nn.ReLU
    dropout_rate: float = 0.1

    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False

    intercept: bool = True
    feature_dropout: float = 0.0
    interaction_degree: Optional[int] = None

    monotone_transform: str = "softplus"
    min_increment: float = 0.00