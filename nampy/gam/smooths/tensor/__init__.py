from .t2 import AlternativeTensorProductSplineTerm
from .te import TensorProductSplineTerm
from .ti import InteractionTensorProductSplineTerm

te = TensorProductSplineTerm
ti = InteractionTensorProductSplineTerm
t2 = AlternativeTensorProductSplineTerm

__all__ = [
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "AlternativeTensorProductSplineTerm",
    "te",
    "ti",
    "t2",
]
