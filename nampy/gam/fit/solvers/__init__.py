"""
Fit solver backends for GAMs.

Three backends are provided, all independent of criterion / optimiser / post-processing:

- :mod:`gaussian_exact`: closed-form penalized least-squares for Gaussian families.
- :mod:`pirls_core` + :mod:`pirls`: penalized IRLS for non-Gaussian families.
- :mod:`penalized_irls`: low-level penalized IRLS entry point (mgcv ``gam.fit3`` analogue)
  used for parity testing against R's mgcv output.
"""

from .gaussian_exact import solve_gaussian_fit
from .pirls import solve_pirls_fit
from .pirls_core import fit_pirls_core
from .penalized_irls import (
    PenalizedIrlsControl,
    fit_penalized_irls,
    fit_penalized_irls_from_model,
)

__all__ = [
    "solve_gaussian_fit",
    "solve_pirls_fit",
    "fit_pirls_core",
    "PenalizedIrlsControl",
    "fit_penalized_irls",
    "fit_penalized_irls_from_model",
]
