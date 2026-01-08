"""Configuration classes for NAMpy models.

This module contains configuration dataclasses for all models in NAMpy.
These configurations define hyperparameters and model settings.
"""

from .nam_config import DefaultNAMConfig
from .nbm_config import DefaultNBMConfig
from .natt_config import DefaultNATTConfig
from .namformer_config import DefaultNAMformerConfig
from .linreg_config import DefaultLinRegConfig
from .boostednam_config import DefaultBoostedNAMConfig
from .snam_config import DefaultSNAMConfig
from .nodegam_config import DefaultNodeGAMConfig

# Note: GPNAM and QNAM use DefaultNAMConfig, so they don't have separate configs
# If gpnam_config.py or qnam_config.py exist in the future, import them here

__all__ = [
    "DefaultNAMConfig",
    "DefaultNBMConfig",
    "DefaultNATTConfig",
    "DefaultNAMformerConfig",
    "DefaultLinRegConfig",
    "DefaultBoostedNAMConfig",
    "DefaultSNAMConfig",
    "DefaultNodeGAMConfig",
]

