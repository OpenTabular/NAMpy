"""Neutral penalty-domain contracts shared by smooths and compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PenaltySpec:
    """Penalty metadata emitted by runtime terms and consumed by the compiler."""

    matrix: np.ndarray
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False
    sp_mode: str | None = None
    sp_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["PenaltySpec"]
