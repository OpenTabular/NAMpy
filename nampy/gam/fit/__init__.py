from .core import fit_model_core
from .backends import available_fit_backends, resolve_fit_backend, solve_fit
from .gaussian import solve_gaussian_fit
from .pirls import solve_pirls_fit
from .covariance import (
    build_bayes_and_freq_covariances,
    select_covariance_matrix,
)
from .offsets import (
    coerce_offset_array,
    resolve_prediction_offset,
)
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
]
