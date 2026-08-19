from dataclasses import dataclass

from ._transformer_base import TransformerConfigBase


@dataclass
class DefaultNAMformerConfig(TransformerConfigBase):
    """
    Configuration for the default NAMformer model.

    Shared transformer hyperparameters: see :class:`TransformerConfigBase`.
    """
