"""
Fit solver backends for GAMs.

Three backends are provided, all independent of criterion / optimiser / post-processing:

- :mod:`gaussian_exact`: closed-form penalized least-squares for Gaussian families.
- :mod:`irls_core` + :mod:`pirls`: shared penalized IRLS core for Gaussian and
  non-Gaussian families.
"""

from .gaussian_exact import solve_gaussian_fit
from .general_family.fixed_smoothing import solve_general_family_fit
from .irls_core import fit_irls_from_model, irls_core
from .pirls import solve_pirls_fit

__all__ = [
    "solve_gaussian_fit",
    "solve_general_family_fit",
    "solve_pirls_fit",
    "irls_core",
    "fit_irls_from_model",
]
