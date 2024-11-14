from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultBayesianNNConfig:
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
    hidden_layer_sizes : list, default=(128, 128, 32)
        Sizes of the hidden layers in the MLP.
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

    # lr: float = 1e-04
    # lr_patience: int = 10
    # weight_decay: float = 1e-06
    # lr_factor: float = 0.1

    hidden_layer_sizes: list = (64,) # (128, 128, 32)
    activation: callable = nn.ReLU()

    # skip_layers: bool = False
    # dropout: float = 0.1
    # norm: str = None
    # use_glu: bool = False
    # skip_connections: bool = False
    # batch_norm: bool = False
    # layer_norm: bool = False
    # interaction_degree: int = None
    # intercept: bool = True
    # feature_dropout: float = 0.0

    gamma_prior_shape: float = 0.5
    gamma_prior_scale: float = 1.0

    gaussian_prior_location: float = 0.0
    gaussian_prior_scale: float = 10.0

