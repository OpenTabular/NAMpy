from .t2 import TensorANOVASplineTerm
from .te import TensorProductSplineTerm
from .ti import InteractionTensorProductSplineTerm

# Backward-compatible short constructor names.
te = TensorProductSplineTerm
ti = InteractionTensorProductSplineTerm
t2 = TensorANOVASplineTerm

__all__ = [
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "TensorANOVASplineTerm",
    "te",
    "ti",
    "t2",
]
