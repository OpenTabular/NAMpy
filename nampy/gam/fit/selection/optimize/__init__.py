"""
Outer optimization of log smoothing parameters (L-BFGS-B and Newton variants).

Submodules: ``basics``, ``objectives``, ``newton``, ``newton_strict``,
``bfgs_strict``, ``efs_strict``, ``trace``, and ``driver``.
"""

from ...smoothing_params import expand_smoothing_params_from_log
from .driver import (
    n_free_smoothing_params,
    optimize_smoothing_params,
    resolve_smoothing_method,
    supports_smoothing_method,
)

__all__ = [
    "expand_smoothing_params_from_log",
    "n_free_smoothing_params",
    "optimize_smoothing_params",
    "resolve_smoothing_method",
    "supports_smoothing_method",
]
