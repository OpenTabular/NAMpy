from dataclasses import dataclass

from .treenam_config import DefaultTreeNAMConfig


@dataclass
class DefaultEnsembleTreeNAMConfig(DefaultTreeNAMConfig):
    """
    Configuration for an ensemble of TreeNAM learners.

    This reuses all TreeNAM hyperparameters and adds only the number of
    ensemble members and the aggregation rule.
    """

    num_estimators: int = 5
    aggregation: str = "mean"  # currently only "mean" is supported
