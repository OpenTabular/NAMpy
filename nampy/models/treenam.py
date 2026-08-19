from ..neural.configs.treenam_config import DefaultTreeNAMConfig
from ..neural.modules.treenam import TreeNAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class TreeNAMRegressor(NeuralRegressor):
    """TreeNAM regressor."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)


class TreeNAMClassifier(NeuralClassifier):
    """TreeNAM classifier."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)


class TreeNAMLSS(NeuralLSS):
    """TreeNAM for distributional regression."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)
