from ..neural.configs.snam_config import DefaultSNAMConfig
from ..neural.modules.snam import SNAM
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class SNAMRegressor(NeuralRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)


class SNAMClassifier(NeuralClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)


class SNAMLSS(NeuralLSS):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)
