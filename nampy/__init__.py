"""NAMpy: Interpretable (Additive) Tabular Deep Learning.

NAMpy is a Python package for neural additive models and related architectures,
offering regression, classification, and distributional regression capabilities
with a scikit-learn compatible interface.
"""

from . import api, models, neural
from .__version__ import __version__

# Import key classes for convenience
from .models import (
    GPNAMLSS,
    NAMLSS,
    NATTLSS,
    NBMLSS,
    QNAM,
    SNAMLSS,
    EnsembleTreeNAMClassifier,
    EnsembleTreeNAMLSS,
    EnsembleTreeNAMRegressor,
    GPNAMClassifier,
    GPNAMRegressor,
    LinRegClassifier,
    LinRegLSS,
    LinRegRegressor,
    NAMClassifier,
    NAMformerClassifier,
    NAMformerLSS,
    NAMformerRegressor,
    NAMRegressor,
    NATTClassifier,
    NATTRegressor,
    NBMClassifier,
    NBMRegressor,
    NodeGAMClassifier,
    NodeGAMLSS,
    NodeGAMRegressor,
    SNAMClassifier,
    SNAMRegressor,
    SplineNAMRegressor,
    TreeNAMClassifier,
    TreeNAMLSS,
    TreeNAMRegressor,
)

__all__ = [
    # Submodules
    "api",
    "models",
    "neural",
    # Main model classes
    "NAMRegressor",
    "NAMClassifier",
    "NAMLSS",
    "GPNAMRegressor",
    "GPNAMClassifier",
    "GPNAMLSS",
    "NBMRegressor",
    "NBMClassifier",
    "NBMLSS",
    "NATTRegressor",
    "NATTClassifier",
    "NATTLSS",
    "NAMformerRegressor",
    "NAMformerClassifier",
    "NAMformerLSS",
    "LinRegRegressor",
    "LinRegClassifier",
    "LinRegLSS",
    "TreeNAMRegressor",
    "TreeNAMClassifier",
    "TreeNAMLSS",
    "EnsembleTreeNAMRegressor",
    "EnsembleTreeNAMClassifier",
    "EnsembleTreeNAMLSS",
    "SNAMRegressor",
    "SNAMClassifier",
    "SNAMLSS",
    "SplineNAMRegressor",
    "NodeGAMRegressor",
    "NodeGAMClassifier",
    "NodeGAMLSS",
    "QNAM",
    # Version
    "__version__",
]
