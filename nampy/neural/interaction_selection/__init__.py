"""Generic feature-interaction detection and hierarchical selection."""

from .archipelago import ArchipelagoDetector
from .combinatorics import interaction_frontier
from .contracts import (
    FeatureGroups,
    InteractionDetector,
    InteractionScore,
    InteractionSelectionResult,
    concatenate_feature_tensors,
)
from .reference import ReferenceMLP, fit_reference_model
from .search import InteractionSearchConfig, select_interactions

__all__ = [
    "ArchipelagoDetector",
    "FeatureGroups",
    "InteractionDetector",
    "InteractionScore",
    "InteractionSearchConfig",
    "InteractionSelectionResult",
    "ReferenceMLP",
    "concatenate_feature_tensors",
    "fit_reference_model",
    "interaction_frontier",
    "select_interactions",
]
