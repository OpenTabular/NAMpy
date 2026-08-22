"""Torch model architectures.

Shared building blocks live in ``nampy.neural.architectures.components``.
"""

from .components import BaseModel
from .ensemble_treenam import EnsembleTreeNAM
from .gpnam import GPNAM
from .igann import IGANN
from .linreg import LinReg
from .nam import NAM
from .namformer import NAMformer
from .natt import NATT
from .nbm import NBM
from .nbm_spam import NBMSPAM
from .nodegam import NodeGAM
from .qnam import QNAM
from .sian import SIAN
from .snam import SNAM
from .spam import SPAM
from .spline_nam import SplineNAM
from .treenam import TreeNAM

__all__ = [
    "BaseModel",
    "EnsembleTreeNAM",
    "GPNAM",
    "IGANN",
    "LinReg",
    "NAM",
    "NAMformer",
    "NATT",
    "NBM",
    "NBMSPAM",
    "NodeGAM",
    "QNAM",
    "SIAN",
    "SNAM",
    "SPAM",
    "SplineNAM",
    "TreeNAM",
]
