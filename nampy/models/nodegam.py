from ..basemodels.nodegam import NodeGAM, NodeGAMLSSBase
from ..configs.nodegam_config import DefaultNodeGAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class NodeGAMRegressor(SklearnBaseRegressor):
    """Scikit-learn wrapper for NODE-GAM regression.

    Accepts fields from DefaultNodeGAMConfig and the shared NAMpy preprocessor
    parameters. Training, prediction, evaluation, and feature-value extraction
    are inherited from SklearnBaseRegressor.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAM, config=DefaultNodeGAMConfig, **kwargs)


class NodeGAMClassifier(SklearnBaseClassifier):
    """Scikit-learn wrapper for NODE-GAM classification.

    Accepts fields from DefaultNodeGAMConfig and the shared NAMpy preprocessor
    parameters. Binary and multiclass output sizing is handled by
    SklearnBaseClassifier and TaskModel.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAM, config=DefaultNodeGAMConfig, **kwargs)


class NodeGAMLSS(SklearnBaseLSS):
    """Scikit-learn wrapper for NODE-GAMLSS distributional regression.

    This uses the upstream-style architecture: one independent NODE-GAM head per
    distribution parameter, with family selection handled by SklearnBaseLSS.fit.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NodeGAMLSSBase, config=DefaultNodeGAMConfig, **kwargs)
