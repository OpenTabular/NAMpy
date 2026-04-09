"""
GAM fitting subsystem (Stage 6 of the pipeline).

Public entry points
-------------------
:func:`fit_model_core`
    Orchestrates the full fit: compile → optimise smoothing → solve → store.

:func:`solve_fit`
    Dispatch to the appropriate backend (Gaussian or P-IRLS) at fixed smoothing
    parameters.

:func:`solve_gaussian_fit`, :func:`solve_pirls_fit`
    Backend-specific solvers.

Key internal types
------------------
:class:`FitCoreSolution`
    Common return type from all backends, passed to :func:`assign_fit_solution`.

Lower-level components (exposed for testing and post-processing)
----------------------------------------------------------------
:func:`fit_penalized_irls`, :func:`fit_penalized_irls_from_model`
    Low-level penalized IRLS entry points used for mgcv parity testing.
:func:`gaussian_smoothness_postprocess`
    Post-fit Gaussian smoothness scores and derivatives.
:func:`pls_fit1_nonneg_w`, :func:`solve_gaussian_penalized_ls_stacked_qr`
    Low-level stacked-QR penalized least-squares solvers.
"""

from .backends import available_fit_backends, resolve_fit_backend, solve_fit
from .covariance import (
    build_bayes_and_freq_covariances,
    select_covariance_matrix,
)
from .linalg.stacked_qr import (
    STACKED_QR_RANK_TOLERANCE,
    gaussian_design_needs_stacked_qr_fit,
    pls_fit1_nonneg_w,
    snap_coef_to_reference_null_space,
    solve_gaussian_penalized_ls_stacked_qr,
)
from .offsets import (
    coerce_offset_array,
    resolve_prediction_offset,
)
from .orchestrator import fit_model_core
from .postprocess.gaussian_smoothness_postprocess import (
    gaussian_smoothness_postprocess,
    merge_gaussian_smoothness_into_fit_result,
)
from .solvers.gaussian_exact import solve_gaussian_fit
from .solvers.penalized_irls import (
    PenalizedIrlsControl,
    fit_penalized_irls,
    fit_penalized_irls_from_model,
)
from .solvers.pirls import solve_pirls_fit
from .state import (
    FitCoreSolution,
    assign_fit_solution,
    compute_edf_by_term,
)

__all__ = [
    "fit_model_core",
    "available_fit_backends",
    "resolve_fit_backend",
    "solve_fit",
    "solve_gaussian_fit",
    "solve_pirls_fit",
    "build_bayes_and_freq_covariances",
    "select_covariance_matrix",
    "coerce_offset_array",
    "resolve_prediction_offset",
    "FitCoreSolution",
    "assign_fit_solution",
    "compute_edf_by_term",
    "PenalizedIrlsControl",
    "fit_penalized_irls",
    "fit_penalized_irls_from_model",
    "gaussian_smoothness_postprocess",
    "merge_gaussian_smoothness_into_fit_result",
    "STACKED_QR_RANK_TOLERANCE",
    "gaussian_design_needs_stacked_qr_fit",
    "pls_fit1_nonneg_w",
    "snap_coef_to_reference_null_space",
    "solve_gaussian_penalized_ls_stacked_qr",
]
