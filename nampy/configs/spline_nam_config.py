from dataclasses import dataclass
from typing import Optional


@dataclass
class DefaultSplineNAMConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    dropout: float = 0.1
    interaction_degree: Optional[int] = None
    intercept: bool = True
    feature_dropout: float = 0.0
    smoothing: float = 0.0
    identify: bool = True
    learn_knots: bool = False
    n_knots: int = 12

