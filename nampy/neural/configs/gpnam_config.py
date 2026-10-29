from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DefaultGPNAMConfig:
    """Configuration for Gaussian Process Neural Additive Models.

    ``kernel_width="auto"`` mirrors the released implementation's
    feature-wise ``std(x) / 24`` rule on the fitted architecture inputs.
    FixedDesignTrainingConfig selects the reference regularized least-squares
    path for ordinary regression; classification and LSS use gradient training.
    """

    kernel_width: float | str = "auto"
    kernel_widths: Optional[Sequence[float]] = None
    rff_num_feat: int = 100
    rff_scheme: str = "quasi_random"
    rff_random_state: Optional[int] = None

    intercept: bool = True

    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, str]]] = None
