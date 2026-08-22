"""Configuration for Sparse Interaction Additive Networks."""

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

import torch.nn as nn


@dataclass
class DefaultSIANConfig:
    """SIAN architecture, sparse search, and reference-network controls."""

    lr: float = 5e-3
    lr_patience: int = 10
    weight_decay: float = 0.0
    lr_factor: float = 0.1
    lr_schedule: str = "plateau"
    optimizer: str = "adagrad"
    layer_sizes: List[int] = field(default_factory=lambda: [16, 12, 8])
    activation: Any = nn.ReLU
    dropout: float = 0.0
    interactions: Optional[Sequence[tuple[str, ...]]] = None
    interaction_degree: Optional[int] = None
    intercept: bool = True
    feature_output_bias: bool = True
    l1_regularization: float = 5e-5
    execution_mode: str = "block_masked"
    residual_network: bool = False
    residual_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 128, 64]
    )

    interaction_detector: str = "archipelago"
    max_interaction_order: int = 3
    interaction_thresholds: float | Mapping[int, float] = 0.1
    threshold_mode: str = "fraction"
    heredity_fraction: float = 0.5
    max_candidates: int = 1000
    max_terms_per_order: Optional[int] = None

    archipelago_baseline: str = "pairwise"
    selection_max_samples: int = 128
    selection_max_pairs: int = 1024
    selection_batch_size: int = 1024
    selection_output_index: int = 0
    reference_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 128, 64]
    )
    reference_epochs: int = 100
    reference_batch_size: int = 128
    reference_lr: float = 5e-3
    reference_weight_decay: float = 0.0
    reference_device: str = "cpu"


__all__ = ["DefaultSIANConfig"]
