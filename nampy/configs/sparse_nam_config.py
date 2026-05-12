from dataclasses import dataclass

from .nam_config import DefaultNAMConfig


@dataclass
class DefaultSparseNAMConfig(DefaultNAMConfig):
    """
    Default config for SparseNAM models.

    This reuses NAM's architecture/configuration and adds the group-lasso controls.
    """

    group_lasso_lambda: float = 0.0
    group_lasso_include_interactions: bool = True
