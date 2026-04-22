from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nampy.gam.linalg import (
    balanced_penalty_template_sqrt_for_rank,
    chol_solve_pivoted,
    column_space_projector,
    compute_preconditioned_inverse,
    constant_null_space_shrinkage,
    geometric_null_space_shrinkage,
    matrix_self_gram,
    matrix_sqrt_psd,
    numerical_rank,
    positive_semidefinite_root,
    project_coef_onto_row_space,
    r_matrix_norm_max_abs,
    r_matrix_norm_one,
    safe_pivoted_cholesky,
    snap_coef_to_reference_null_space,
    symmetric_eigen_partition,
    symmetric_eigh,
    symmetrize_from_lower_triangle,
    symmetrize_matrix,
    upper_triangular_rrank,
)


def test_linalg_basic_exports_match_expected_behavior():
    """Verify that linalg basic exports match expected behavior."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(matrix_self_gram(X), X @ X.T, atol=1e-12)
    np.testing.assert_allclose(
        column_space_projector(X),
        X @ np.linalg.pinv(X),
        atol=1e-12,
    )
    assert r_matrix_norm_one(X) == 7.0
    assert r_matrix_norm_max_abs(X) == 4.0
    np.testing.assert_allclose(
        symmetrize_matrix(np.array([[1.0, 3.0], [0.0, 2.0]], dtype=np.float64)),
        np.array([[1.0, 1.5], [1.5, 2.0]], dtype=np.float64),
        atol=1e-12,
    )


def test_linalg_pivoted_cholesky_helpers_solve_spd_system():
    """Verify that linalg pivoted cholesky helpers solve SPD system."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    chol_upper, piv, ipiv, ok = safe_pivoted_cholesky(
        A,
        np.eye(2, dtype=np.float64) * np.finfo(np.float64).eps ** 0.5,
    )

    assert ok
    np.testing.assert_allclose(
        chol_solve_pivoted(chol_upper, rhs, piv=piv, ipiv=ipiv),
        np.linalg.solve(A, rhs),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_preconditioned_inverse(
            chol_upper,
            np.ones(2, dtype=np.float64),
            2,
            piv=piv,
            ipiv=ipiv,
        ),
        np.linalg.inv(A),
        atol=1e-12,
    )


def test_linalg_eigen_and_rank_exports_match_expected_behavior():
    """Verify that linalg eigen and rank exports match expected behavior."""
    penalty = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    evals, evecs = symmetric_eigh(penalty, descending=True)
    assert evals[0] >= evals[1]
    np.testing.assert_allclose(
        evecs @ np.diag(evals) @ evecs.T,
        penalty,
        atol=1e-12,
    )

    dec = symmetric_eigen_partition(np.diag([4.0, 0.0]).astype(np.float64))
    assert dec["rank"] == 1
    assert dec["null_space_dim"] == 1

    sqrt_psd = matrix_sqrt_psd(np.diag([4.0, 0.0]).astype(np.float64))
    np.testing.assert_allclose(
        sqrt_psd @ sqrt_psd.T,
        np.diag([4.0, 0.0]).astype(np.float64),
        atol=1e-12,
    )

    root = positive_semidefinite_root(np.diag([9.0, 1.0]).astype(np.float64), rank=1)
    assert root.shape == (2, 1)
    np.testing.assert_allclose(root @ root.T, np.diag([9.0, 0.0]), atol=1e-12)

    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    coef = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    ref = np.array([2.0, 2.0, -4.0], dtype=np.float64)
    projected = project_coef_onto_row_space(X, coef)
    snapped = snap_coef_to_reference_null_space(coef, X, ref)
    np.testing.assert_allclose(X @ projected, X @ coef, atol=1e-12)
    np.testing.assert_allclose(X @ snapped, X @ coef, atol=1e-12)
    np.testing.assert_allclose(snapped, ref, atol=1e-12)
    assert numerical_rank(X) == 2

    penalty_blocks = [
        SimpleNamespace(
            matrix=np.array([[2.0]], dtype=np.float64),
            coef_slice=slice(0, 1),
            smoothing_index=0,
        )
    ]
    np.testing.assert_allclose(
        balanced_penalty_template_sqrt_for_rank(
            penalty_blocks,
            fit_intercept=False,
            n_coef=1,
        ),
        np.array([[1.0]], dtype=np.float64),
        atol=1e-12,
    )
    assert upper_triangular_rrank(
        np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    ) == 1

    lower_sym = symmetrize_from_lower_triangle(
        np.array([[1.0, 9.0], [2.0, 3.0]], dtype=np.float64)
    )
    np.testing.assert_allclose(
        lower_sym,
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
        atol=1e-12,
    )

    geom = geometric_null_space_shrinkage(
        np.diag([4.0, 0.0, 0.0]).astype(np.float64),
        shrink=0.1,
        descending=True,
    )
    geom_eigs = np.linalg.eigvalsh(geom)
    np.testing.assert_allclose(
        geom_eigs,
        np.array([0.04, 0.4, 4.0], dtype=np.float64),
        atol=1e-12,
    )

    const = constant_null_space_shrinkage(
        np.diag([4.0, 0.0, 0.0]).astype(np.float64),
        shrink=0.1,
    )
    const_eigs = np.linalg.eigvalsh(const)
    np.testing.assert_allclose(
        const_eigs,
        np.array([0.4, 0.4, 4.0], dtype=np.float64),
        atol=1e-12,
    )
