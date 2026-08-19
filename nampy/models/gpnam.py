from ..neural.configs.gpnam_config import DefaultGPNAMConfig
from ..neural.modules.gpnam import GPNAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class GPNAMRegressor(NeuralRegressor):
    """Gaussian Process Neural Additive Model regressor."""

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)


class GPNAMClassifier(NeuralClassifier):
    """Gaussian Process Neural Additive Model classifier."""

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)


class GPNAMLSS(NeuralLSS):
    """
    GP-NAM wrapped for LSS-style output heads.

    This is a package extension beyond the paper's main regression/binary
    classification focus.
    """

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)
