from dataclasses import dataclass
import torch.nn as nn
import jax.numpy as jnp
import jax.nn as jnn


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
    num_epochs : int, default=5000
        Number of epochs for which to train the model.
    gamma_prior_shape : float, default=0.5
        Shape parameter for the Gamma prior.
    gamma_prior_scale : float, default=1.0
        Scale parameter for the Gamma prior.
    gaussian_prior_location : float, default=0.0
        Location parameter for the Gaussian prior.
    gaussian_prior_scale : float, default=5.0
        Scale parameter for the Gaussian prior.
    """

    # Model definition parameters.
    hidden_layer_sizes: list = (128, 128, 64) # (10, 10, 10, 10, 10)
    # activation: callable = nn.SELU()
    activation: str = "selu"

    # skip_layers: bool = False
    dropout: float = 0.0
    # norm: str = None
    use_glu: bool = False
    # skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = True

    gamma_prior_shape: float = 0.5
    gamma_prior_scale: float = 1.0

    gaussian_prior_location: float = 0.0
    gaussian_prior_scale: float = 5.0

    # Optimization parameters.
    num_epochs: int = 250  # 25000

    lr: float = 1e-4
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    mcmc_step_size: float = 2.0

