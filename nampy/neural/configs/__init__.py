"""Configuration classes for NAMpy models.

This module contains configuration dataclasses for all models in NAMpy.
These configurations define hyperparameters and model settings.
"""

from .ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from .gpnam_config import DefaultGPNAMConfig
from .igann_config import DefaultIGANNConfig
from .linreg_config import DefaultLinRegConfig
from .nam_config import DefaultNAMConfig
from .namformer_config import DefaultNAMformerConfig
from .natt_config import DefaultNATTConfig
from .nbm_config import DefaultNBMConfig
from .nbm_spam_config import DefaultNBMSPAMConfig
from .nodegam_config import DefaultNodeGAMConfig
from .qnam_config import DefaultQNAMConfig
from .sian_config import DefaultSIANConfig
from .snam_config import DefaultSNAMConfig
from .spam_config import DefaultSPAMConfig
from .spline_nam_config import DefaultSplineNAMConfig
from .treenam_config import DefaultTreeNAMConfig

__all__ = [
    "DefaultNAMConfig",
    "DefaultSNAMConfig",
    "DefaultNBMConfig",
    "DefaultNBMSPAMConfig",
    "DefaultNATTConfig",
    "DefaultNAMformerConfig",
    "DefaultLinRegConfig",
    "DefaultGPNAMConfig",
    "DefaultIGANNConfig",
    "DefaultQNAMConfig",
    "DefaultSIANConfig",
    "DefaultSplineNAMConfig",
    "DefaultTreeNAMConfig",
    "DefaultEnsembleTreeNAMConfig",
    "DefaultNodeGAMConfig",
    "DefaultSPAMConfig",
]
