from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultNodeGAMConfig:
    """
    Configuration class for the default NodeGAM with predefined hyperparameters.

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
    arch : str, default="GAM"
        Architecture type. Choose between "GAM" or "GAMAtt".
    num_trees : int, default=200
        Number of trees in each layer.
    num_layers : int, default=2
        Number of layers of trees.
    depth : int, default=3
        Depth of each tree.
    addi_tree_dim : int, default=0
        Additional dimension for the outputs of each tree.
    output_dropout : float, default=0.0
        Dropout rate on the output of each tree.
    last_dropout : float, default=0.3
        Dropout rate on the weight of the last linear layer.
    colsample_bytree : float, default=0.5
        The random proportion of features allowed in each tree.
    l2_lambda : float, default=0.0
        L2 penalty coefficient on the outputs of trees.
    dim_att : int, default=8
        Dimension of the attention embedding (only used in GAMAtt).
    anneal_steps : int, default=2000
        Number of steps for temperature annealing.
    interaction_degree : int, default=1
        Degree of interactions to be modeled. If >= 2, enables GA2M.
    feature_dropout : float, default=0.0
        Dropout rate for feature regularization.
    """

    # Optimization parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Architecture parameters
    arch: str = "GAM"
    num_trees: int = 200
    num_layers: int = 2
    depth: int = 3
    addi_tree_dim: int = 0
    output_dropout: float = 0.0
    last_dropout: float = 0.3
    colsample_bytree: float = 0.5
    l2_lambda: float = 0.0
    dim_att: int = 8
    anneal_steps: int = 2000

    # Model parameters
    interaction_degree: int = 2
    feature_dropout: float = 0.0
    quantile_preprocessing: str = "feature"
    quantile_noise: float = 0.0
    quantile_output_distribution: str = "normal"
    quantile_n_quantiles: int = 2000
