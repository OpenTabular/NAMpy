from ..basemodels.treenam import TreeNAM
from ..configs.treenam_config import DefaultTreeNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class TreeNAMRegressor(SklearnBaseRegressor):
    """TreeNAM regressor."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)


class TreeNAMClassifier(SklearnBaseClassifier):
    """TreeNAM classifier."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)


class TreeNAMLSS(SklearnBaseLSS):
    """TreeNAM for distributional regression."""

    def __init__(self, **kwargs):
        super().__init__(model=TreeNAM, config=DefaultTreeNAMConfig, **kwargs)
