"""Configuration classes for NAMpy models.

This module contains configuration dataclasses for all models in NAMpy.
These configurations define hyperparameters and model settings.
"""

from .linreg_config import DefaultLinRegConfig
from .nam_config import DefaultNAMConfig
from .namformer_config import DefaultNAMformerConfig
from .natt_config import DefaultNATTConfig
from .nbm_config import DefaultNBMConfig
from .ngboost_config import DefaultNGBoostConfig
from .nodegam_config import DefaultNodeGAMConfig
from .spline_nam_config import DefaultSplineNAMConfig

# Note: GPNAM and QNAM use DefaultNAMConfig, so they don't have separate configs
# If gpnam_config.py or qnam_config.py exist in the future, import them here

__all__ = [
    "DefaultNAMConfig",
    "DefaultNBMConfig",
    "DefaultNGBoostConfig",
    "DefaultNATTConfig",
    "DefaultNAMformerConfig",
    "DefaultLinRegConfig",
    "DefaultSplineNAMConfig",
    "DefaultNodeGAMConfig",
]
