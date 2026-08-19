from ..neural.configs.linreg_config import DefaultLinRegConfig
from ..neural.modules.linreg import LinReg
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class LinRegRegressor(NeuralRegressor):
    """Linear-effects additive model regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultLinRegConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=LinReg, config=DefaultLinRegConfig, **kwargs)


class LinRegClassifier(NeuralClassifier):
    """Linear-effects additive model classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultLinRegConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=LinReg, config=DefaultLinRegConfig, **kwargs)


class LinRegLSS(NeuralLSS):
    """Linear-effects additive model for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultLinRegConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=LinReg, config=DefaultLinRegConfig, **kwargs)
