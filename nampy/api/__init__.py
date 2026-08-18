"""Shared public contracts for NAMpy model backends."""

from .capabilities import Capabilities, SupportsCapabilities
from .persistence import PersistableModel
from .results import AdditivePrediction
from .schema import FeatureSchema

__all__ = [
    "FeatureSchema",
    "AdditivePrediction",
    "Capabilities",
    "SupportsCapabilities",
    "PersistableModel",
]
