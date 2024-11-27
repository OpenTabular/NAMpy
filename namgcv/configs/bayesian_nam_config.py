from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultBayesianNAMConfig:
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
    interaction_degree: int = 2
    intercept: bool = True
    feature_dropout: float = 0.0

    # Note: The prior parameters are only used in the joint optimization procedure.
    gamma_prior_shape: float = 0.5
    gamma_prior_scale: float = 1.0

    gaussian_prior_location: float = 0.0
    gaussian_prior_scale: float = 5.0

    # Optimization parameters.
    num_epochs: int = 5000

    lr: float = 1e-4
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    mcmc_step_size: float = 5.0
