from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultBayesianNNConfig:
    """
    Configuration class for the default NAM with predefined hyperparameters.

    Parameters
    ----------
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
    gaussian_prior_location : float, default=0.0
        Location parameter for the Gaussian prior.
    gaussian_prior_scale : float, default=5.0
        Scale parameter for the Gaussian prior.
    """

    # Model definition parameters.
    hidden_layer_sizes: list = (1000, 500, 100, 50, 25)
    # activation: callable = nn.SELU()
    activation: str = "relu"

    # skip_layers: bool = False
    dropout: float = 0.25
    # norm: str = None
    use_glu: bool = False
    # skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = True

    # Prior parameters - covariance matrix for non-isotropic Gaussian prior.
    use_hierarchical_priors: bool = False
    use_correlated_biases: bool = False
    use_correlated_weights: bool = False
    lkj_concentration: float = 1.0

    # Weight prior parameters (Isotropic Gaussian).
    gaussian_prior_location: float = 0.0
    gaussian_prior_scale: float = 10.0

    # Weight prior scale parameter hyperprior (Half-Normal).
    w_layer_scale_half_normal_hyperscale: float = 1.0
    b_layer_scale_half_normal_hyperscale: float = 1.0