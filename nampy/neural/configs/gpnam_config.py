from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DefaultGPNAMConfig:
    """Configuration for Gaussian Process Neural Additive Models.

    ``kernel_width="auto"`` mirrors the released implementation's
    feature-wise ``std(x) / 24`` rule on the fitted architecture inputs.
    ``solver="cg"`` selects the reference regularized least-squares path for
    ordinary regression; classification and LSS continue through the shared
    objective-driven training engine.
    """

    lr: float = 1e-2
    lr_patience: int = 10
    weight_decay: float = 0.0
    lr_factor: float = 0.1
    lr_schedule: str = "inverse_sqrt"

    kernel_width: float | str = "auto"
    kernel_widths: Optional[Sequence[float]] = None
    rff_num_feat: int = 100
    rff_scheme: str = "quasi_random"
    rff_random_state: Optional[int] = None

    intercept: bool = True
    solver: str = "cg"
    ridge: float = 0.05
    cg_rtol: float = 1e-6
    cg_max_iter: Optional[int] = None

    interaction_degree: Optional[int] = None
    interactions: Optional[Sequence[tuple[str, str]]] = None
