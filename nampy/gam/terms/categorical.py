"""Categorical/random-effect/factor-smooth terms."""

from ..smooths.categorical.categorical_utils import (
    all_bool_like,
    as_object_1d,
    factor_indicator_matrix,
    is_factor_like_vector,
    stable_unique_levels,
    try_numeric_1d,
)
from ..smooths.categorical.factor_smooth import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
)
from ..smooths.categorical.mrf import MarkovRandomFieldTerm
from ..smooths.categorical.random_effect import RandomEffectTerm

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
]
