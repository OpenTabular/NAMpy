from dataclasses import dataclass
import torch.nn as nn


@dataclass(frozen=True)
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

    train_val_split_ratio: float = 0.8

    # Model definition parameters.
    interaction_degree: int = 1
    intercept: bool = False
    num_mixture_components: int = 1
    feature_dropout: float = 0.0

    intercept_prior_shape: float = 0.0
    intercept_prior_scale: float = 2.0

    # Sigma is only sampled if we are doing mean regression.
    sigma_prior_scale: float = 100.0

    # Optimization parameters.
    mcmc_step_size: float = 1.0
    num_chains: int = 10
    num_samples = 1000
    num_warmup_samples = 100
    target_accept_prob: float = 0.8

    # Deep ensemble parameters.
    use_deep_ensemble: bool = False
    de_num_epochs: int = 2000
    de_lr: float = 1e-4
    de_lr_transition_steps: int = 200
    de_lr_decay: float = 0.9
    de_lr_staircase: bool = True
    warm_start_early_stop_patience: int = 10