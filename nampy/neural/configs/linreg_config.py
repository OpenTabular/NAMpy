from dataclasses import dataclass


@dataclass
class DefaultLinRegConfig:
    """
    Configuration class for the default LinReg with predefined hyperparameters.

    Parameters
    ----------
    intercept : bool
        Whether to use a learnable intercept parameter.
    """

    intercept: bool = True
