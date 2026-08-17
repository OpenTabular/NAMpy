from __future__ import annotations

import numpy as np

from nampy.gam.linalg import (
    column_space_projector,
    covariance_standard_errors,
    matrix_self_gram,
    row_space_projector,
    symmetric_spectrum,
)
from tests.mgcv_invariant_policy import (
    final_fit_uses_exact_orientation_parity,
    gam_setup_uses_invariant_transform,
    gam_side_uses_invariant_transform,
    preoptimization_blocks_align_basis_columns,
    preoptimization_blocks_compare_range_root_representation,
    stable_column_space_projector,
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


def test_stable_column_space_projector_matches_rotation_invariant_projector():
    """Verify the shared stable projector keeps the same column-space invariant."""
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

    np.testing.assert_allclose(
        stable_column_space_projector(X),
        stable_column_space_projector(X @ rotation),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        stable_column_space_projector(X),
        column_space_projector(X),
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


def test_invariant_policy_centralizes_non_unique_representation_surfaces():
    """Verify shared policy marks the current non-unique surfaces consistently."""
    assert gam_setup_uses_invariant_transform("gaussian_tp_two_dim")
    assert not gam_setup_uses_invariant_transform("gaussian_two_cr")

    assert gam_side_uses_invariant_transform("tprs.smooth")
    assert gam_side_uses_invariant_transform("fs.interaction")
    assert not gam_side_uses_invariant_transform("cr.smooth")

    assert preoptimization_blocks_align_basis_columns("gaussian_tp_two_dim")
    assert not preoptimization_blocks_align_basis_columns("gaussian_two_cr")
    assert not preoptimization_blocks_compare_range_root_representation(
        "gaussian_fs"
    )
    assert preoptimization_blocks_compare_range_root_representation(
        "gaussian_two_cr"
    )

    class _Term:
        def __init__(self, basis_name):
            self.basis_name = basis_name

    class _Compiled:
        def __init__(self, *basis_names):
            self.compiled_terms = [_Term(name) for name in basis_names]

    class _Model:
        def __init__(self, *basis_names):
            self.compiled_model_ = _Compiled(*basis_names)

    assert final_fit_uses_exact_orientation_parity(
        _Model("cr"), skip_coef_comparison=False
    )
    assert not final_fit_uses_exact_orientation_parity(
        _Model("tp"), skip_coef_comparison=False
    )
    assert not final_fit_uses_exact_orientation_parity(
        _Model("fs"), skip_coef_comparison=False
    )
    assert not final_fit_uses_exact_orientation_parity(
        _Model("cr"), skip_coef_comparison=True
    )
