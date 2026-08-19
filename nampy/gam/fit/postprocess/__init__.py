"""Post-fit diagnostics and score/derivative computation.

The lazy ``__getattr__`` below is cycle-breaking, not decorative:
``fit.state`` imports this package while ``fit.backends`` is still
initializing, and ``gaussian_smoothness_postprocess`` imports
``selection.criteria.dispatch`` which imports ``fit.backends`` back.
"""

from __future__ import annotations

__all__ = [
    "gaussian_smoothness_postprocess",
    "merge_gaussian_smoothness_into_fit_result",
]


def __getattr__(name):
    if name in {
        "gaussian_smoothness_postprocess",
        "merge_gaussian_smoothness_into_fit_result",
    }:
        from .gaussian_smoothness_postprocess import (
            gaussian_smoothness_postprocess,
            merge_gaussian_smoothness_into_fit_result,
        )

        if name == "gaussian_smoothness_postprocess":
            return gaussian_smoothness_postprocess
        return merge_gaussian_smoothness_into_fit_result
    raise AttributeError(name)
