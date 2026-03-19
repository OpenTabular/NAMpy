from ..basemodels.ensemble_treenam import EnsembleTreeNAM
from ..configs.ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class EnsembleTreeNAMRegressor(SklearnBaseRegressor):
    """Simple ensemble of TreeNAM learners for regression."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )


class EnsembleTreeNAMClassifier(SklearnBaseClassifier):
    """Simple ensemble of TreeNAM learners for classification."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )


class EnsembleTreeNAMLSS(SklearnBaseLSS):
    """Simple ensemble of TreeNAM learners for distributional regression."""

    def __init__(self, **kwargs):
        super().__init__(
            model=EnsembleTreeNAM,
            config=DefaultEnsembleTreeNAMConfig,
            **kwargs,
        )