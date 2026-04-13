"""
Outer optimization of log smoothing parameters (L-BFGS-B, Newton, indefinite-Hessian Newton for P-IRLS).

Submodules: ``basics``, ``objectives``, ``outer``, ``driver``.
"""

from ..criteria import criterion_infinite_sp_signal
from .driver import (
    expand_smoothing_params_from_log,
    n_free_smoothing_params,
    optimize_smoothing_params,
    resolve_smoothing_method,
    supports_smoothing_method,
)

__all__ = [
    "criterion_infinite_sp_signal",
    "expand_smoothing_params_from_log",
    "n_free_smoothing_params",
    "optimize_smoothing_params",
    "resolve_smoothing_method",
    "supports_smoothing_method",
]
