from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DefaultTreeNAMConfig:
    """
    Configuration for TreeNAM.
    """

    tree_depth: int = 4
    tree_lamda: float = 1e-3
    tree_temperature: float = 1.0
    use_hard_routing_in_eval: bool = False

    feature_dropout: float = 0.0
    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, ...]]] = None
    intercept: bool = True
