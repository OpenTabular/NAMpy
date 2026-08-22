from dataclasses import dataclass


@dataclass
class DefaultLinRegConfig:
    """
    Configuration class for the default LinReg with predefined hyperparameters.

    Parameters
    ----------
    lr : float
        Learning rate for the optimizer.
    lr_patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float
        Factor by which the learning rate will be reduced.
    intercept : bool
        Whether to use a learnable intercept parameter.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    intercept: bool = True
