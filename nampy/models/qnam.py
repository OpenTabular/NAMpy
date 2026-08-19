from ..neural.configs.qnam_config import DefaultQNAMConfig
from ..neural.modules.qnam import QNAM
from .lss import NeuralLSS


class QNAMLSS(NeuralLSS):
    """Quantile Neural Additive Model (always ``family='quantile'``).

    Hyperparameters: see :class:`nampy.neural.configs.DefaultQNAMConfig` plus
    shared preprocessing options in :class:`NeuralEstimatorBase`.
    """

    def __init__(self, **kwargs):
        family = kwargs.pop("family", "quantile")
        if str(family).lower() != "quantile":
            raise ValueError("QNAMLSS only supports family='quantile'.")
        distributional_kwargs = kwargs.pop("distributional_kwargs", None)
        if distributional_kwargs is None:
            distributional_kwargs = {"quantiles": [0.25, 0.5, 0.75]}
        super().__init__(
            model=QNAM,
            config=DefaultQNAMConfig,
            family=family,
            distributional_kwargs=distributional_kwargs,
            **kwargs,
        )
