from ..basemodels.snam import SNAM
from ..configs.snam_config import DefaultSNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class SNAMRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)


class SNAMClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)


class SNAMLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=SNAM, config=DefaultSNAMConfig, **kwargs)
