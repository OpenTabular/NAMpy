"""Public estimator family generated from the QNAM declaration."""

from ._registered import estimator_family

_family = estimator_family("qnam", module_name=__name__)
QNAMLSS = _family.lss

__all__ = ["QNAMLSS"]
