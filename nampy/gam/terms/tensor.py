"""Tensor-product smooth term implementations."""

from ..smooths.tensor.t2 import TensorANOVASplineTerm
from ..smooths.tensor.te import TensorProductSplineTerm
from ..smooths.tensor.ti import InteractionTensorProductSplineTerm

__all__ = [
    "TensorProductSplineTerm",
    "InteractionTensorProductSplineTerm",
    "TensorANOVASplineTerm",
]
