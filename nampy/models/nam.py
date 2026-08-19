# models/nam.py
from ..neural.configs.nam_config import DefaultNAMConfig
from ..neural.modules.nam import NAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class NAMRegressor(NeuralRegressor):
    """Neural Additive Model regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)


class NAMClassifier(NeuralClassifier):
    """Neural Additive Model classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)


class NAMLSS(NeuralLSS):
    """Neural Additive Model for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)
