"""Shared linear algebra helpers for GAM internals.

Generic, reusable matrix/norm/factorization utilities live here so parity
surfaces and solver code can share one canonical implementation.
"""

from .cholesky import (
    chol_solve_pivoted,
    compute_preconditioned_inverse,
    pivoted_cholesky,
    safe_pivoted_cholesky,
)
from .eigen import (
    matrix_sqrt_psd,
    positive_semidefinite_root,
    symmetric_eigen_partition,
    symmetric_eigh,
    symmetric_eigvalsh,
)
from .matrix import symmetrize_matrix
from .norms import r_matrix_norm_max_abs, r_matrix_norm_one
from .rank import (
    balanced_penalty_template_sqrt_for_rank,
    matrix_is_rank_deficient,
    numerical_rank,
    project_coef_onto_row_space,
    snap_coef_to_reference_null_space,
    svd_null_space_basis,
    symmetric_penalty_rank,
    upper_triangular_condition_indicator,
    upper_triangular_rrank,
)
from .shrinkage import (
    constant_null_space_shrinkage,
    geometric_null_space_shrinkage,
    symmetrize_from_lower_triangle,
)
from .subspaces import (
    column_space_projector,
    covariance_standard_errors,
    matrix_self_gram,
    matrix_summary,
    row_space_projector,
    symmetric_spectrum,
)

__all__ = [
    "symmetrize_matrix",
    "r_matrix_norm_one",
    "r_matrix_norm_max_abs",
    "safe_pivoted_cholesky",
    "chol_solve_pivoted",
    "compute_preconditioned_inverse",
    "pivoted_cholesky",
    "symmetric_eigh",
    "symmetric_eigvalsh",
    "symmetric_eigen_partition",
    "matrix_sqrt_psd",
    "positive_semidefinite_root",
    "numerical_rank",
    "matrix_is_rank_deficient",
    "svd_null_space_basis",
    "project_coef_onto_row_space",
    "snap_coef_to_reference_null_space",
    "balanced_penalty_template_sqrt_for_rank",
    "symmetric_penalty_rank",
    "upper_triangular_condition_indicator",
    "upper_triangular_rrank",
    "symmetrize_from_lower_triangle",
    "geometric_null_space_shrinkage",
    "constant_null_space_shrinkage",
    "matrix_self_gram",
    "column_space_projector",
    "row_space_projector",
    "symmetric_spectrum",
    "matrix_summary",
    "covariance_standard_errors",
]
