"""Configuration for the Scalable Polynomial Additive Model."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DefaultSPAMConfig:
    """Defaults for the released SPAM parameterization.

    ``ranks[i]`` is the low-rank width for polynomial degree ``i + 2``.  The
    upstream constructor has no usable rank default; NAMpy chooses 100 for a
    practical quadratic model while preserving the upstream degree mapping.
    """

    lr: float = 1e-3
    lr_patience: int = 10
    weight_decay: float = 0.0
    lr_factor: float = 0.1
    ranks: List[int] = field(default_factory=lambda: [100])
    dropout: float = 0.0
    ignore_unary: bool = False
    reg_order: int = 2
    lower_order_correction: bool = False
    use_geometric_mean: bool = True
    orthogonal: bool = False
    proximal: bool = False
    regularization_scale: float = 0.0
    basis_l1_regularization: float = 0.0
    intercept: bool = True
