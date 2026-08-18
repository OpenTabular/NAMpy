from dataclasses import dataclass
from typing import Optional


@dataclass
class DefaultTreeNAMConfig:
    """
    Configuration for TreeNAM.
    """

    lr: float = 1e-3
    lr_patience: int = 10
    weight_decay: float = 1e-6
    lr_factor: float = 0.1

    tree_depth: int = 4
    tree_lamda: float = 1e-3
    tree_temperature: float = 1.0
    use_hard_routing_in_eval: bool = False

    feature_dropout: float = 0.0
    interaction_degree: Optional[int] = None
    intercept: bool = True
