from ..basemodels.gpnam import GPNAM
from ..configs.gpnam_config import DefaultGPNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class GPNAMRegressor(SklearnBaseRegressor):
    """Gaussian Process Neural Additive Model regressor."""

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)


class GPNAMClassifier(SklearnBaseClassifier):
    """Gaussian Process Neural Additive Model classifier."""

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)


class GPNAMLSS(SklearnBaseLSS):
    """
    GP-NAM wrapped for LSS-style output heads.

    This is a package extension beyond the paper's main regression/binary
    classification focus.
    """

    def __init__(self, **kwargs):
        super().__init__(model=GPNAM, config=DefaultGPNAMConfig, **kwargs)
