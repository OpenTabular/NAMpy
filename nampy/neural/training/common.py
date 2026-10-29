"""Training plans, tensor conversion and reproducible initialization."""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _training_tensor(value, *, device=None):
    """Convert training inputs without sharing read-only NumPy storage."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, device=device)


@dataclass
class TrainingPlan:
    """Task-specific wiring for the shared neural training engine.

    Attributes
    ----------
    objective : NeuralObjective
        Output, target, loss, and metric semantics forwarded to
        :class:`TaskModule` independently of the architecture.
    stratify : array-like or None
        Stratification labels for the automatic train/validation split.
    """

    objective: Any
    stratify: Any = None

    @property
    def datamodule_regression(self) -> bool:
        """Derive the data-label representation from the objective."""
        return bool(self.objective.datamodule_regression)


def seed_neural_runtime(seed: int) -> None:
    """Seed model construction and data-order RNGs from the public fit seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
