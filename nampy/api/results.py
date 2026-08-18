"""Backend-neutral result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AdditivePrediction:
    """Prediction outputs and additive contributions on the link scale."""

    response: np.ndarray
    link: np.ndarray
    terms: dict[str, np.ndarray]
    intercept: float | np.ndarray
    backend: Literal["gam", "neural"]
    offset: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"gam", "neural"}:
            raise ValueError("backend must be either 'gam' or 'neural'.")
