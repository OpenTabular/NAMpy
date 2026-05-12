from ..basemodels.sparse_nam import SparseNAM
from ..configs.sparse_nam_config import DefaultSparseNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class SparseNAMRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=SparseNAM, config=DefaultSparseNAMConfig, **kwargs)


class SparseNAMClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=SparseNAM, config=DefaultSparseNAMConfig, **kwargs)


class SparseNAMLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=SparseNAM, config=DefaultSparseNAMConfig, **kwargs)
