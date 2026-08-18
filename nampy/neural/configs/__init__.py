"""Configuration classes for NAMpy models.

This module contains configuration dataclasses for all models in NAMpy.
These configurations define hyperparameters and model settings.
"""

from .ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from .gpnam_config import DefaultGPNAMConfig
from .linreg_config import DefaultLinRegConfig
from .nam_config import DefaultNAMConfig
from .namformer_config import DefaultNAMformerConfig
from .natt_config import DefaultNATTConfig
from .nbm_config import DefaultNBMConfig
from .nodegam_config import DefaultNodeGAMConfig
from .qnam_config import DefaultQNAMConfig
from .snam_config import DefaultSNAMConfig
from .spline_nam_config import DefaultSplineNAMConfig
from .treenam_config import DefaultTreeNAMConfig

__all__ = [
    "DefaultNAMConfig",
    "DefaultSNAMConfig",
    "DefaultNBMConfig",
    "DefaultNATTConfig",
    "DefaultNAMformerConfig",
    "DefaultLinRegConfig",
    "DefaultGPNAMConfig",
    "DefaultQNAMConfig",
    "DefaultSplineNAMConfig",
    "DefaultTreeNAMConfig",
    "DefaultEnsembleTreeNAMConfig",
    "DefaultNodeGAMConfig",
]
