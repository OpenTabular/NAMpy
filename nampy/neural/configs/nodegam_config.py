from dataclasses import dataclass


@dataclass
class DefaultNodeGAMConfig:
    """
    Configuration class for the default NodeGAM with predefined hyperparameters.

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
    arch : str
        Architecture type. Choose between "GAM" or "GAMAtt".
    num_trees : int
        Number of trees in each layer.
    num_layers : int
        Number of layers of trees.
    depth : int
        Depth of each tree.
    addi_tree_dim : int
        Additional dimension for the outputs of each tree.
    output_dropout : float
        Dropout rate on the output of each tree.
    input_dropout : float
        Dropout applied to each tree layer's input independently.
    last_dropout : float
        Dropout rate on the weight of the last linear layer.
    colsample_bytree : float
        The random proportion of features allowed in each tree.
    l2_lambda : float
        L2 penalty coefficient on the outputs of trees.
    dim_att : int
        Dimension of the attention embedding (only used in GAMAtt).
    anneal_steps : int
        Number of steps for temperature annealing.
    selector_activation : str
        Feature-selector activation: ``"entmax15"`` or ``"sparsemax"``.
    bin_activation : str
        Tree-bin activation: ``"entmoid15"`` or ``"sparsemoid"``.
    interaction_degree : int
        Degree of interactions to be modeled. If >= 2, enables GA2M.
    feature_dropout : float
        Dropout rate for feature regularization.
    l2_interactions : float
        L2 penalty applied only to learned pairwise interaction outputs.
    l1_interactions : float
        L1 penalty applied only to learned pairwise interaction outputs.
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
    input_dropout: float = 0.0
    last_dropout: float = 0.3
    colsample_bytree: float = 0.5
    l2_lambda: float = 0.0
    dim_att: int = 8
    anneal_steps: int = 2000
    selector_activation: str = "entmax15"
    bin_activation: str = "entmoid15"

    # Model parameters
    interaction_degree: int = 2
    feature_dropout: float = 0.0
    l2_interactions: float = 0.0
    l1_interactions: float = 0.0
