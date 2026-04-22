from __future__ import annotations

import numpy as np

from nampy.gam.linalg import (
    column_space_projector,
    covariance_standard_errors,
    matrix_self_gram,
    row_space_projector,
    symmetric_spectrum,
)


def test_matrix_space_invariants_ignore_column_sign_flips():
    """Verify that matrix space invariants ignore column sign flips."""
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, -4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )
    X_flipped = X * np.array([1.0, -1.0], dtype=np.float64)

    np.testing.assert_allclose(matrix_self_gram(X), matrix_self_gram(X_flipped))
    np.testing.assert_allclose(
        column_space_projector(X),
        column_space_projector(X_flipped),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        row_space_projector(X),
        row_space_projector(X_flipped),
        atol=1e-12,
    )


def test_matrix_space_invariants_ignore_orthogonal_basis_rotation():
    """Verify that matrix space invariants ignore orthogonal basis rotation."""
    X = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.5],
            [2.0, -1.0],
            [-0.5, 0.25],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    X_rotated = X @ rotation

    np.testing.assert_allclose(
        column_space_projector(X),
        column_space_projector(X_rotated),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        row_space_projector(X),
        row_space_projector(X_rotated),
        atol=1e-12,
    )


def test_covariance_standard_errors_and_spectrum_are_stable_helpers():
    """Verify that covariance standard errors and spectrum are stable helpers."""
    cov = np.array([[4.0, 1.5], [1.5, 9.0]], dtype=np.float64)
    permuted = cov[::-1, ::-1]

    np.testing.assert_allclose(
        covariance_standard_errors(cov),
        np.array([2.0, 3.0], dtype=np.float64),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        symmetric_spectrum(cov),
        symmetric_spectrum(permuted),
        atol=1e-12,
    )
