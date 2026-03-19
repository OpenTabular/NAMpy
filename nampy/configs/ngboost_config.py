from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DefaultNGBoostConfig:
    """Configuration for NGBoost-based models."""

    distribution: Optional[Any] = None
    score: str = "logscore"
    base_learner: str = "tree"
    base_learner_kwargs: Dict[str, Any] = field(default_factory=dict)
    natural_gradient: bool = True
    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 1.0
    col_sample: float = 1.0
    verbose: bool = True
    verbose_eval: int = 100
    tol: float = 1e-4
    random_state: Optional[int] = None
    validation_fraction: float = 0.1
    early_stopping_rounds: Optional[int] = None
