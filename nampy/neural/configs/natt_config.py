from dataclasses import dataclass

from ._transformer_base import TransformerConfigBase


@dataclass
class DefaultNATTConfig(TransformerConfigBase):
    """
    Configuration for the default NATT model.

    Shared transformer hyperparameters: see :class:`TransformerConfigBase`.

    Parameters
    ----------
    layer_sizes : tuple
        Sizes of the layers in the per-feature MLPs.
    skip_layers : bool
        Whether to skip layers in the per-feature MLPs.
    """

    layer_sizes: tuple = (128, 128, 32)
    skip_layers: bool = False
