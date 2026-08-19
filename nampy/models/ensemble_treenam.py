from ..neural.configs.ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from ..neural.modules.ensemble_treenam import EnsembleTreeNAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class EnsembleTreeNAMRegressor(NeuralRegressor):
    """Simple ensemble of TreeNAM learners for regression."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )


class EnsembleTreeNAMClassifier(NeuralClassifier):
    """Simple ensemble of TreeNAM learners for classification."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )


class EnsembleTreeNAMLSS(NeuralLSS):
    """Simple ensemble of TreeNAM learners for distributional regression."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )
