"""Compatibility re-exports for post-optimization heuristics."""

from .heuristics.rollback import (
    _accept_flat_boundary_result,
    _accept_tiny_step_line_search_result,
    _criterion_infinite_sp_signal,
    _preserve_optimize_result_metadata,
    _rollback_working_infinite_smoothing_params,
)
from .heuristics.stabilize import (
    _collapse_near_zero_smoothing_params,
    _coordinate_refine_smoothing_params,
    _refine_null_space_smoothing_params,
    _snap_gaussian_random_effect_boundary,
    _stabilize_factor_smooth_shared_ridge,
    _stabilize_flat_smoothing_params,
    _stabilize_joint_negbin_flat_ridge,
)

__all__ = [
    "_accept_flat_boundary_result",
    "_accept_tiny_step_line_search_result",
    "_collapse_near_zero_smoothing_params",
    "_coordinate_refine_smoothing_params",
    "_criterion_infinite_sp_signal",
    "_preserve_optimize_result_metadata",
    "_refine_null_space_smoothing_params",
    "_rollback_working_infinite_smoothing_params",
    "_snap_gaussian_random_effect_boundary",
    "_stabilize_factor_smooth_shared_ridge",
    "_stabilize_flat_smoothing_params",
    "_stabilize_joint_negbin_flat_ridge",
]
