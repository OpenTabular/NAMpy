"""End-to-end mgcv snapshot parity for the core model matrix."""

from _mgcv_snapshot_parity_shared import TestMgcvParity as _SharedTestMgcvParity


class TestMgcvParity:
    test_gaussian_fixed_sp_matches_mgcv_exactly = (
        _SharedTestMgcvParity.test_gaussian_fixed_sp_matches_mgcv_exactly
    )
    test_gaussian_re_fixed_sp_matches_mgcv_exactly = (
        _SharedTestMgcvParity.test_gaussian_re_fixed_sp_matches_mgcv_exactly
    )
    test_te_reml_smoothing_parameters_match_mgcv = (
        _SharedTestMgcvParity.test_te_reml_smoothing_parameters_match_mgcv
    )
    test_gaussian_re_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_re_reml_matches_mgcv
    )
    test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv
    )
    test_gaussian_fs_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_fs_reml_matches_mgcv
    )
    test_gaussian_sz_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_sz_reml_matches_mgcv
    )
    test_gaussian_mrf_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_mrf_reml_matches_mgcv
    )
    test_gaussian_concurvity_full_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_concurvity_full_matches_mgcv
    )
    test_poisson_concurvity_full_matches_mgcv = (
        _SharedTestMgcvParity.test_poisson_concurvity_full_matches_mgcv
    )
    test_gaussian_concurvity_pairwise_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_concurvity_pairwise_matches_mgcv
    )
    test_poisson_concurvity_pairwise_matches_mgcv = (
        _SharedTestMgcvParity.test_poisson_concurvity_pairwise_matches_mgcv
    )
    test_gaussian_te_fixed_sp_matches_mgcv_exactly = (
        _SharedTestMgcvParity.test_gaussian_te_fixed_sp_matches_mgcv_exactly
    )
    test_gaussian_ti_fixed_sp_matches_mgcv_exactly = (
        _SharedTestMgcvParity.test_gaussian_ti_fixed_sp_matches_mgcv_exactly
    )
    test_gaussian_t2_fixed_sp_matches_mgcv_exactly = (
        _SharedTestMgcvParity.test_gaussian_t2_fixed_sp_matches_mgcv_exactly
    )
    test_gaussian_te_reml_multi_penalty_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_te_reml_multi_penalty_matches_mgcv
    )
    test_gaussian_ti_reml_multi_penalty_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_ti_reml_multi_penalty_matches_mgcv
    )
    test_gaussian_t2_reml_multi_penalty_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_t2_reml_multi_penalty_matches_mgcv
    )
    test_gaussian_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_reml_matches_mgcv
    )
    test_gaussian_reml_default_s_basis_matches_mgcv_tp_default = (
        _SharedTestMgcvParity.test_gaussian_reml_default_s_basis_matches_mgcv_tp_default
    )
    test_gaussian_reml_sig2_rss_match_mgcv_two_cr_smooths = (
        _SharedTestMgcvParity.test_gaussian_reml_sig2_rss_match_mgcv_two_cr_smooths
    )
    test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth = (
        _SharedTestMgcvParity.test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth
    )
    test_gaussian_reml_sig2_matches_mgcv_joint_outer_mrf_exact = (
        _SharedTestMgcvParity.test_gaussian_reml_sig2_matches_mgcv_joint_outer_mrf_exact
    )
    test_binomial_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_binomial_reml_matches_mgcv
    )
    test_poisson_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_poisson_reml_matches_mgcv
    )
    test_gaussian_select_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_select_reml_matches_mgcv
    )
    test_gamma_reml_matches_mgcv = _SharedTestMgcvParity.test_gamma_reml_matches_mgcv
    test_gamma_reml_optimizes_without_abnormal_warning = (
        _SharedTestMgcvParity.test_gamma_reml_optimizes_without_abnormal_warning
    )
    test_negbin_reml_matches_mgcv = _SharedTestMgcvParity.test_negbin_reml_matches_mgcv
    test_poisson_reml_with_formula_offset_matches_mgcv = (
        _SharedTestMgcvParity.test_poisson_reml_with_formula_offset_matches_mgcv
    )
    test_optimized_tensor_t2_snapshot_matches_mgcv = (
        _SharedTestMgcvParity.test_optimized_tensor_t2_snapshot_matches_mgcv
    )
    test_gaussian_mrf_low_rank_reml_matches_mgcv = (
        _SharedTestMgcvParity.test_gaussian_mrf_low_rank_reml_matches_mgcv
    )
