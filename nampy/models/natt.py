from ..neural.configs.natt_config import DefaultNATTConfig
from ..neural.modules.natt import NATT
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class NATTRegressor(NeuralRegressor):
    """Neural Additive Tabular Transformer regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNATTConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NATT, config=DefaultNATTConfig, **kwargs)


class NATTClassifier(NeuralClassifier):
    """Neural Additive Tabular Transformer classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNATTConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NATT, config=DefaultNATTConfig, **kwargs)


class NATTLSS(NeuralLSS):
    """Neural Additive Tabular Transformer for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNATTConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NATT, config=DefaultNATTConfig, **kwargs)
