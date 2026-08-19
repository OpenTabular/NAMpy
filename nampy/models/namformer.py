from ..neural.configs.namformer_config import DefaultNAMformerConfig
from ..neural.modules.namformer import NAMformer
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


class NAMformerRegressor(NeuralRegressor):
    """NAMformer (transformer-augmented neural additive model) regressor.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMformerConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAMformer, config=DefaultNAMformerConfig, **kwargs)


class NAMformerClassifier(NeuralClassifier):
    """NAMformer (transformer-augmented neural additive model) classifier.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMformerConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAMformer, config=DefaultNAMformerConfig, **kwargs)


class NAMformerLSS(NeuralLSS):
    """NAMformer (transformer-augmented neural additive model) for distributional regression.

    Hyperparameters: see :class:`nampy.neural.configs.DefaultNAMformerConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAMformer, config=DefaultNAMformerConfig, **kwargs)
