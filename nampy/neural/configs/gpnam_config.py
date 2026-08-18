from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DefaultGPNAMConfig:
    """
    Default config for Gaussian Process Neural Additive Models (GP-NAM).

    Notes
    -----
    - kernel_width is the scalar fallback used for every input dimension if
      kernel_widths is not provided.
    - kernel_widths, when provided, must match the number of scalar post-
      preprocessing input dimensions.
    """

    lr: float = 1e-4
    lr_patience: int = 10
    weight_decay: float = 1e-6
    lr_factor: float = 0.1

    kernel_width: float = 0.2
    kernel_widths: Optional[List[float]] = None
    rff_num_feat: int = 100

    intercept: bool = True
    use_deterministic_rff_grid: bool = True
