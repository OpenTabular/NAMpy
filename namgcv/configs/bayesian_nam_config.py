from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultBayesianNAMConfig:
    """
    Configuration class for the default NAM with predefined hyperparameters.

    Parameters
    ----------
    interaction_degree : int, default=1
        Degree of the interaction terms.
    intercept : bool, default=True
        Whether to include a global intercept term.
    feature_dropout : float, default=0.0
        Dropout rate for the input features.
    intercept_prior_shape : float, default=0.0
        Shape parameter for the prior on the intercept.
    intercept_prior_scale : float, default=1.0
        Scale parameter for the prior on the intercept.
    gaussian_prior_location : float, default=0.0
        Location parameter for the Gaussian prior.
    gaussian_prior_scale : float, default=5.0
        Scale parameter for the Gaussian prior.
    """

    # Model definition parameters.
    interaction_degree: int = 1
    intercept: bool = True
    feature_dropout: float = 0.0

    intercept_prior_shape: float = 0.0
    intercept_prior_scale: float = 1.0

    gaussian_prior_location: float = 0.0
    gaussian_prior_scale: float = 1.0

    # Optimization parameters.
    mcmc_step_size: float = 1.0
    num_chains: int = 1
    num_samples = 1000
    target_accept_prob: float = 0.8