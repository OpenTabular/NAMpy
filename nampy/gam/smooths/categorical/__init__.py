from .categorical_utils import (
    all_bool_like,
    as_object_1d,
    factor_indicator_matrix,
    is_factor_like_vector,
    stable_unique_levels,
    try_numeric_1d,
)
from .fs import FSmoothInteractionTerm, SZSmoothInteractionTerm
from .mrf import MarkovRandomFieldTerm
from .re import RandomEffectTerm

fs = FSmoothInteractionTerm
sz = SZSmoothInteractionTerm
mrf = MarkovRandomFieldTerm
re = RandomEffectTerm

__all__ = [
    "as_object_1d",
    "try_numeric_1d",
    "all_bool_like",
    "is_factor_like_vector",
    "stable_unique_levels",
    "factor_indicator_matrix",
    "RandomEffectTerm",
    "FSmoothInteractionTerm",
    "SZSmoothInteractionTerm",
    "MarkovRandomFieldTerm",
    "fs",
    "sz",
    "mrf",
    "re",
]
