"""Focused Gaussian weighted-fit and random-effect regression parity coverage."""

from _mgcv_snapshot_parity_shared import TestMgcvParity as _SharedTestMgcvParity


class TestGaussianRandomEffectRegressions:
    test_gaussian_re_coef_full_exact_mgcv_noisy_interior_sp = (
        _SharedTestMgcvParity.test_gaussian_re_coef_full_exact_mgcv_noisy_interior_sp
    )
    test_gaussian_re_gam_snapshot_row_space_coef_matches_mgcv_near_singular = (
        _SharedTestMgcvParity.test_gaussian_re_gam_snapshot_row_space_coef_matches_mgcv_near_singular
    )
    test_project_coef_onto_row_space_preserves_fitted_values = (
        _SharedTestMgcvParity.test_project_coef_onto_row_space_preserves_fitted_values
    )
    test_snap_coef_to_reference_null_space_matches_reference = (
        _SharedTestMgcvParity.test_snap_coef_to_reference_null_space_matches_reference
    )
