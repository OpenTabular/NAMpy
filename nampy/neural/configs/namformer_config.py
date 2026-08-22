from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch.nn as nn


@dataclass
class DefaultNAMformerConfig:
    """
    Configuration for the default NAMformer model.

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
    d_model : int
        Dimensionality of the model.
    n_layers : int
        Number of layers in the transformer.
    n_heads : int
        Number of attention heads in the transformer.
    attn_dropout : float
        Dropout rate for the attention mechanism.
    ff_dropout : float
        Dropout rate for the feed-forward layers.
    activation : callable
        Activation function for the feature MLPs.
    dropout : float
        Dropout rate for regularization.
    norm : str
        Normalization method to be used.
    use_glu : bool
        Whether to use Gated Linear Units (GLU).
    skip_connections : bool
        Whether to use skip connections.
    batch_norm : bool
        Whether to use batch normalization.
    layer_norm : bool
        Whether to use layer normalization.
    embedding_activation : callable
        Activation function for embeddings.
    head_layer_sizes : tuple
        Sizes of the layers in the head of the model.
    head_dropout : float
        Dropout rate for the head layers.
    head_skip_layers : bool
        Whether to skip layers in the head.
    head_activation : callable
        Activation function for the head layers.
    head_use_batch_norm : bool
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool
        Whether to apply layer normalization after embedding.
    pooling_method : str
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool
        Whether to apply normalization before other operations in each transformer block.
    bias : bool
        Whether to use bias in the linear layers.
    transformer_activation : callable
        Activation function for the transformer layers.
    layer_norm_eps : float
        Epsilon value for layer normalization.
    transformer_dim_feedforward : int
        Dimensionality of the feed-forward layers in the transformer.
    cat_encoding : str
        Whether to use integer encoding or one-hot encoding for cat features.
    use_cls : bool
        Whether to prepend a CLS token to the feature sequence.
    intercept : bool
        Whether to fit a global intercept.
    interaction_degree : int or None
        Maximum degree of feature interactions; None disables interactions.
    feature_dropout : float
        Dropout rate applied to whole feature contributions.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    attn_dropout: float = 0.2
    ff_dropout: float = 0.1
    activation: Any = field(default_factory=nn.ReLU)
    dropout: float = 0.1
    norm: str = "LayerNorm"
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    embedding_activation: Any = field(default_factory=nn.Identity)
    head_layer_sizes: tuple = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Any = field(default_factory=nn.SELU)
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "avg"
    norm_first: bool = True
    bias: bool = True
    transformer_activation: Any = field(default_factory=nn.ReLU)
    layer_norm_eps: float = 1e-05
    transformer_dim_feedforward: int = 256
    cat_encoding: str = "int"
    use_cls: bool = True
    intercept: bool = True
    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, ...]]] = None
    feature_dropout: float = 0.0
