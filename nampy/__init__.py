"""NAMpy: Interpretable (Additive) Tabular Deep Learning.

NAMpy is a Python package for neural additive models and related architectures,
offering regression, classification, and distributional regression capabilities
with a scikit-learn compatible interface.
"""

from . import basemodels, configs, data_utils, models, preprocessing, utils
from .__version__ import __version__

# Import key classes for convenience
from .models import (
    GPNAMLSS,
    NAMLSS,
    NATTLSS,
    NBMLSS,
    QNAM,
    GAMRegressor,
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
    SNAMRegressor,
    TreeNAMRegressor,
)

__all__ = [
    # Submodules
    "basemodels",
    "models",
    "data_utils",
    "preprocessing",
    "utils",
    "configs",
    # Main model classes
    "GAMRegressor",
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
    "SNAMRegressor",
    "NodeGAMRegressor",
    "NodeGAMClassifier",
    "NodeGAMLSS",
    "QNAM",
    # Version
    "__version__",
]
