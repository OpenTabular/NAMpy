"""Neural task semantics and training objectives, including quantile loss."""

from .core import (
    BinaryObjective,
    DistributionObjective,
    MulticlassObjective,
    NeuralObjective,
    RegressionObjective,
    classification_objective,
)
from .parameterized import ParameterizedObjective
from .quantile import Quantile

__all__ = [
    "ParameterizedObjective",
    "BinaryObjective",
    "DistributionObjective",
    "MulticlassObjective",
    "NeuralObjective",
    "RegressionObjective",
    "Quantile",
    "classification_objective",
]
