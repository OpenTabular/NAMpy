from ..neural.configs.nodegam_config import DefaultNodeGAMConfig
from ..neural.modules.nodegam import NodeGAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class NodeGAMRegressor(NeuralRegressor):
    """NodeGAM (oblivious-tree neural GAM) regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNodeGAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAM, config=DefaultNodeGAMConfig, **kwargs)


class NodeGAMClassifier(NeuralClassifier):
    """NodeGAM (oblivious-tree neural GAM) classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNodeGAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAM, config=DefaultNodeGAMConfig, **kwargs)


class NodeGAMLSS(NeuralLSS):
    """NodeGAM (oblivious-tree neural GAM) for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNodeGAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAM, config=DefaultNodeGAMConfig, **kwargs)
