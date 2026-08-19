# models/nbm.py
from ..neural.configs.nbm_config import DefaultNBMConfig
from ..neural.modules.nbm import NBM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class NBMRegressor(NeuralRegressor):
    """Neural Basis Model regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNBMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NBM, config=DefaultNBMConfig, **kwargs)


class NBMClassifier(NeuralClassifier):
    """Neural Basis Model classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNBMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NBM, config=DefaultNBMConfig, **kwargs)


class NBMLSS(NeuralLSS):
    """Neural Basis Model for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNBMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NBM, config=DefaultNBMConfig, **kwargs)
