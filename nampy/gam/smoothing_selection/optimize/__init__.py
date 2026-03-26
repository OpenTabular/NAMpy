"""
Outer optimization of log smoothing parameters (L-BFGS-B, Newton, indefinite-Hessian Newton for P-IRLS).

Submodules: ``basics``, ``objectives``, ``postprocess``, ``outer``, ``driver``.
"""

from ..criteria import criterion_infinite_sp_signal
from .driver import (
    expand_smoothing_params_from_log,
    n_free_smoothing_params,
    optimize_smoothing_params,
    resolve_smoothing_method,
    supports_smoothing_method,
)
from .postprocess import _rollback_working_infinite_smoothing_params

__all__ = [
    "criterion_infinite_sp_signal",
    "_rollback_working_infinite_smoothing_params",
    "expand_smoothing_params_from_log",
    "n_free_smoothing_params",
    "optimize_smoothing_params",
    "resolve_smoothing_method",
    "supports_smoothing_method",
]
