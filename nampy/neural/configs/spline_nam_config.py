from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DefaultSplineNAMConfig:
    """Default configuration for spline-based neural additive models."""

    dropout: float = 0.1
    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, ...]]] = None
    intercept: bool = True
    feature_dropout: float = 0.0
    smoothing: float = 0.0
    identify: bool = True
    learn_knots: bool = False
    n_knots: int = 12
