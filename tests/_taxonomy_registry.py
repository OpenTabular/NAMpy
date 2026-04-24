from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeafCoverageExpectation:
    leaf_id: str
    nodeid_parts: tuple[str, ...]


def _leaf(leaf_id: str, *nodeid_parts: str) -> LeafCoverageExpectation:
    return LeafCoverageExpectation(leaf_id=leaf_id, nodeid_parts=tuple(nodeid_parts))


_SMOOTH_MARK_NAMES = {
    "cr": "smooth_cr",
    "cs": "smooth_cs",
    "cc": "smooth_cc",
    "ps": "smooth_ps",
    "tp": "smooth_tp",
    "ts": "smooth_ts",
    "te": "smooth_te",
    "ti": "smooth_ti",
    "t2": "smooth_t2",
    "gp": "smooth_gp",
    "fs": "smooth_fs",
    "sz": "smooth_sz",
    "mrf": "smooth_mrf",
    "re": "smooth_re",
}

_FAMILY_MARK_NAMES = {
    "gaussian": "family_gaussian",
    "binomial": "family_binomial",
    "poisson": "family_poisson",
    "gamma": "family_gamma",
    "negbin": "family_negbin",
    "gaulss": "family_gaulss",
    "gammals": "family_gammals",
    "gevlss": "family_gevlss",
    "shashlss": "family_shashlss",
    "ziplss": "family_ziplss",
    "general": "family_general",
}

_METHOD_MARK_NAMES = {
    "fixed": "method_fixed",
    "reml": "method_reml",
    "ml": "method_ml",
    "laml": "method_laml",
}

_STATUS_MARKS_BY_FILE = {
    "test_mgcv_known_gaps.py": {"status_known_gap"},
    "test_mgcv_parity_failing_and_warnings.py": {"status_failing_or_warning"},
    "test_gam_mgcv_patch_regressions.py": {"status_regression"},
}

_DEFAULT_MARKS_BY_FILE = {
    "test_mgcv_snapshot_core_matrix.py": {"surface_snapshot"},
    "test_mgcv_snapshot_parity.py": {"surface_snapshot"},
    "test_mgcv_snapshot_extended_matrix.py": {"surface_snapshot"},
    "test_mgcv_pc_id_parity.py": {"surface_snapshot"},
    "test_mgcv_known_gaps.py": {"surface_snapshot"},
    "test_mgcv_parity_failing_and_warnings.py": {"surface_snapshot"},
    "test_mgcv_output_parity.py": {"surface_output"},
    "test_mgcv_smoothcon_parity.py": {"surface_smoothcon"},
    "test_mgcv_raw_constructor_parity.py": {"surface_smoothcon"},
    "test_mgcv_score_hist_trace_parity.py": {"surface_trace"},
    "test_mgcv_optimization_lifecycle_parity.py": {"surface_trace"},
    "test_mgcv_linked_id_trace_parity.py": {"surface_trace"},
    "test_mgcv_inner_trace_parity.py": {"surface_trace"},
    "test_mgcv_newton_parity.py": {"surface_trace"},
    "test_mgcv_newton_exact_parity.py": {"surface_trace"},
    "test_mgcv_k_check_parity.py": {"surface_kcheck"},
    "test_mgcv_score_gamma_parity.py": {"surface_derivatives"},
    "test_gam_gaussian_smoothness_postprocess_parity.py": {
        "surface_derivatives",
        "family_gaussian",
    },
    "test_general_family_mgcv_parity.py": {
        "surface_derivatives",
        "family_general",
    },
    "test_mgcv_gamlss_core.py": {"surface_derivatives", "family_general"},
    "test_mgcv_gamlss_gaulss.py": {"surface_derivatives", "family_gaulss"},
    "test_mgcv_gamlss_gammals.py": {"surface_derivatives", "family_gammals"},
    "test_mgcv_gamlss_gevlss.py": {"surface_derivatives", "family_gevlss"},
    "test_mgcv_gamlss_shashlss.py": {"surface_derivatives", "family_shashlss"},
    "test_mgcv_gamlss_ziplss.py": {"surface_derivatives", "family_ziplss"},
    "test_mgcv_gaussian_backend_selection.py": {
        "surface_backend",
        "family_gaussian",
    },
    "test_gam_mgcv_patch_regressions.py": {"surface_regression"},
    "test_gam_optimization_lifecycle_contracts.py": {"surface_regression"},
    "test_gam_test_suite_contracts.py": {"surface_regression"},
}

_SELECTION_CAPABLE_FILES = {
    "test_mgcv_snapshot_core_matrix.py",
    "test_mgcv_snapshot_parity.py",
    "test_mgcv_snapshot_extended_matrix.py",
    "test_mgcv_pc_id_parity.py",
    "test_mgcv_known_gaps.py",
    "test_mgcv_output_parity.py",
    "test_mgcv_score_hist_trace_parity.py",
    "test_mgcv_linked_id_trace_parity.py",
    "test_mgcv_score_gamma_parity.py",
    "test_general_family_mgcv_parity.py",
}

_PRIMARY_COVERAGE_BY_MARK = {
    "smooth_cr": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/smooths/test_mgcv_pc_id_parity.py",
    ),
    "smooth_cs": (
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
    ),
    "smooth_cc": (
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "tests/smooths/test_mgcv_smoothcon_parity.py",
    ),
    "smooth_ps": (
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "smooth_tp": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "smooth_ts": (
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "smooth_te": (
        "tests/smooths/test_mgcv_te_stage_parity.py",
        "tests/parity/test_mgcv_output_parity.py",
    ),
    "smooth_ti": (
        "tests/smooths/test_mgcv_ti_stage_parity.py",
        "tests/parity/test_mgcv_output_parity.py",
    ),
    "smooth_t2": (
        "tests/smooths/test_mgcv_t2_stage_parity.py",
        "tests/parity/test_mgcv_output_parity.py",
    ),
    "smooth_gp": (
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "smooth_fs": (
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
    ),
    "smooth_sz": (
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
    ),
    "smooth_mrf": (
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
    ),
    "smooth_re": (
        "tests/smooths/test_mgcv_smoothcon_parity.py",
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
    ),
    "family_gaussian": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/optimization/test_mgcv_outer_optimization_parity.py",
    ),
    "family_binomial": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/optimization/test_mgcv_score_hist_trace_parity.py",
    ),
    "family_poisson": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/optimization/test_mgcv_score_hist_trace_parity.py",
        "tests/optimization/test_mgcv_inner_trace_parity.py",
    ),
    "family_gamma": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
    ),
    "family_negbin": (
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "tests/optimization/test_mgcv_inner_trace_parity.py",
        "tests/optimization/test_mgcv_joint_branch_trace_parity.py",
    ),
    "family_gaulss": (
        "tests/families/test_mgcv_gamlss_gaulss.py",
        "tests/optimization/test_mgcv_outer_optimization_parity.py",
    ),
    "family_gammals": ("tests/families/test_mgcv_gamlss_gammals.py",),
    "family_gevlss": ("tests/families/test_mgcv_gamlss_gevlss.py",),
    "family_shashlss": ("tests/families/test_mgcv_gamlss_shashlss.py",),
    "family_ziplss": ("tests/families/test_mgcv_gamlss_ziplss.py",),
    "family_general": (
        "tests/families/test_general_family_mgcv_parity.py",
        "tests/families/test_mgcv_gamlss_core.py",
    ),
    "method_fixed": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/smooths/test_mgcv_pc_id_parity.py",
    ),
    "method_reml": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/optimization/test_mgcv_outer_optimization_parity.py",
    ),
    "method_ml": (
        "tests/optimization/test_mgcv_outer_optimization_parity.py",
        "tests/parity/test_mgcv_snapshot_parity.py",
    ),
    "method_laml": (
        "tests/optimization/test_mgcv_ncv_qncv_parity.py",
        "tests/optimization/test_mgcv_joint_branch_trace_parity.py",
    ),
    "select_true": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "select_false": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/optimization/test_mgcv_score_hist_trace_parity.py",
    ),
    "surface_snapshot": (
        "tests/parity/test_mgcv_snapshot_core_matrix.py",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
    ),
    "surface_output": ("tests/parity/test_mgcv_output_parity.py",),
    "surface_smoothcon": (
        "tests/smooths/test_mgcv_raw_constructor_parity.py",
        "tests/smooths/test_mgcv_smoothcon_parity.py",
    ),
    "surface_trace": (
        "tests/optimization/test_mgcv_score_hist_trace_parity.py",
        "tests/optimization/test_mgcv_optimization_lifecycle_parity.py",
        "tests/optimization/test_mgcv_outer_optimization_parity.py",
        "tests/optimization/test_mgcv_inner_trace_parity.py",
        "tests/smooths/test_mgcv_linked_id_trace_parity.py",
    ),
    "surface_kcheck": ("tests/diagnostics/test_mgcv_k_check_parity.py",),
    "surface_derivatives": (
        "tests/optimization/test_mgcv_score_gamma_parity.py",
        "tests/families/test_general_family_mgcv_parity.py",
    ),
    "surface_regression": (
        "tests/regressions/test_gam_mgcv_patch_regressions.py",
        "tests/regressions/test_gam_optimization_lifecycle_contracts.py",
        "tests/regressions/test_gam_test_suite_contracts.py",
    ),
    "surface_backend": ("tests/optimization/test_mgcv_gaussian_backend_selection.py",),
}


_DIRECT_SUPPORTED_LEAF_EXPECTATIONS = (
    _leaf(
        "snapshot_extended_binomial_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_binomial_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_poisson_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_poisson_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_re_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_re_select_reml_matches_mgcv_exactly",
    ),
    _leaf(
        "snapshot_extended_gaussian_sz_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_sz_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_mrf_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_mrf_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_weighted_poisson_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_weighted_poisson_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_weighted_binomial_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_weighted_binomial_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_fs_ps_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_fs_ps_marginal_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_fs_ps_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_fs_ps_marginal_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_theta_0p5_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_theta_0p5_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_theta_2p0_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_theta_2p0_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_theta_0p5_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_theta_0p5_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_theta_2p0_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_theta_2p0_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_binomial_probit_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_binomial_probit_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_binomial_probit_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_binomial_probit_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_binomial_cloglog_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_binomial_cloglog_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_binomial_cloglog_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_binomial_cloglog_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gamma_inverse_link_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_inverse_link_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gamma_inverse_link_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_inverse_link_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_transformed_formula_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_transformed_formula_fixed_sp_snapshot_matches_mgcv_exactly",
    ),
    _leaf(
        "snapshot_extended_transformed_formula_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_transformed_formula_reml_snapshot_matches_mgcv_exactly",
    ),
    _leaf(
        "snapshot_extended_gamma_identity_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_identity_link_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gamma_identity_reml_prediction",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_identity_link_reml_prediction_matches_mgcv_fixed_sp",
    ),
    _leaf(
        "snapshot_extended_gamma_identity_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_identity_link_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_te_cc_cc_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_te_cc_cc_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_ti_cc_cc_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_ti_cc_cc_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_t2_cc_cc_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_t2_cc_cc_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_te_ts_cr_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_te_ts_cr_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_ti_ts_cr_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_ti_ts_cr_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_te_tp_cr_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_te_tp_cr_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_ti_gp_cr_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_ti_gp_cr_fixed_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_te_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_te_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gamma_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gamma_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_select_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_select_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_fs_numeric_by_fixed",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_fs_numeric_by_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_negbin_estimated_theta_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_negbin_estimated_theta_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_fs_four_levels_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_fs_4levels_reml_matches_mgcv",
    ),
    _leaf(
        "snapshot_extended_gaussian_sz_three_by_three_reml",
        "tests/parity/test_mgcv_snapshot_extended_matrix.py",
        "test_gaussian_sz_3x3_reml_matches_mgcv",
    ),
    _leaf(
        "pc_cr_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cr_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cs_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cs_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_ps_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ps_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cc_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cc_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cr_factor_by_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cr_factor_by_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_tp_multivariate_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_tp_multivariate_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_gp_multivariate_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_gp_multivariate_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_ts_multivariate_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ts_multivariate_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cs_factor_by_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cs_factor_by_pc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cr_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cr_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_cs_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cs_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_cc_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cc_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_ps_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ps_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_ps_factor_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ps_factor_by_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_tp_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_tp_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_ts_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ts_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_tp_numeric_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_tp_numeric_by_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_ts_numeric_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ts_numeric_by_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_gp_numeric_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_gp_numeric_by_pc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_ps_numeric_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_ps_numeric_by_pc_reml_matches_mgcv",
    ),
    _leaf(
        "linked_id_cr_compatible_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_compatible_k_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "linked_id_cr_compatible_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_compatible_k_reml_matches_mgcv",
    ),
    _leaf(
        "linked_id_cr_incompatible_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_incompatible_k_fixed_sp_predictions_match_mgcv",
    ),
    _leaf(
        "linked_id_cr_incompatible_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_incompatible_k_reml_matches_mgcv",
    ),
    _leaf(
        "linked_id_pc_internal_consistency",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_with_pc_fits_and_enforces_constraint",
    ),
    _leaf(
        "pc_cr_factor_by_select_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cr_factor_by_select_reml_matches_mgcv",
    ),
    _leaf(
        "pc_cc_factor_by_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cc_factor_by_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "pc_cc_factor_by_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cc_factor_by_reml_matches_mgcv",
    ),
    _leaf(
        "linked_id_cc_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cc_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "linked_id_cc_reml",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cc_reml_matches_mgcv",
    ),
    _leaf(
        "pc_cr_select_term_sp_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cr_select_with_term_sp_vector_fixed_matches_mgcv",
    ),
    _leaf(
        "pc_cc_select_scalar_sp_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_cc_select_with_scalar_sp_fixed_matches_mgcv",
    ),
    _leaf(
        "linked_id_cr_three_term_fixed",
        "tests/smooths/test_mgcv_pc_id_parity.py",
        "test_linked_cr_three_terms_fixed_sp_matches_mgcv",
    ),
    _leaf(
        "output_anova_gaussian_cr_uni_reml",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_anova_model_comparison",
        "gaussian_cr_uni_reml",
    ),
    _leaf(
        "output_newdata_link_gaussian_cr_uni_reml",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_predictions_and_standard_errors",
        "gaussian_cr_uni_reml",
        "link",
    ),
    _leaf(
        "output_newdata_link_gaussian_cr_uni_fixed",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_predictions_and_standard_errors",
        "gaussian_cr_uni_fixed",
        "link",
    ),
    _leaf(
        "output_newdata_linked_id",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_terms_linked_id",
    ),
    _leaf(
        "output_newdata_lpmatrix_gaussian",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_lpmatrix_gaussian",
    ),
    _leaf(
        "output_transformed_formula_link",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_surfaces",
        "link",
    ),
    _leaf(
        "output_transformed_formula_response",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_surfaces",
        "response",
    ),
    _leaf(
        "output_transformed_formula_lpmatrix",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_surfaces",
        "lpmatrix",
    ),
    _leaf(
        "output_transformed_formula_terms",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_term_surfaces",
        "terms",
    ),
    _leaf(
        "output_transformed_formula_iterms",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_term_surfaces",
        "iterms",
    ),
    _leaf(
        "output_transformed_formula_iterms_type_2",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_newdata_transformed_formula_term_surfaces",
        "iterms_type_2",
    ),
    _leaf(
        "output_transformed_formula_training_standard_errors",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_transformed_formula_training_standard_errors_reml",
    ),
    _leaf(
        "output_fixed_gaussian_offset_predictions",
        "tests/parity/test_mgcv_output_parity.py",
        "test_output_parity_fixed_sp_gaussian_offset_predictions",
    ),
    _leaf(
        "general_stage_select_true_prediction_link",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_select_true_newdata_prediction_surfaces_match_mgcv",
        "link",
    ),
    _leaf(
        "general_stage_select_true_prediction_response",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_select_true_newdata_prediction_surfaces_match_mgcv",
        "response",
    ),
    _leaf(
        "general_stage_select_true_prediction_terms",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_select_true_newdata_prediction_surfaces_match_mgcv",
        "terms",
    ),
    _leaf(
        "general_stage_select_true_unconditional_link",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_unconditional_standard_errors_match_mgcv_on_select_true_case",
        "link",
    ),
    _leaf(
        "general_stage_select_true_unconditional_response",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_unconditional_standard_errors_match_mgcv_on_select_true_case",
        "response",
    ),
    _leaf(
        "general_stage_select_true_unconditional_terms",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_unconditional_standard_errors_match_mgcv_on_select_true_case",
        "terms",
    ),
    _leaf(
        "general_stage_linked_id_public_prediction_link",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_linked_id_public_prediction_surfaces_match_mgcv_on_supported_gaussian_case",
        "link",
    ),
    _leaf(
        "general_stage_linked_id_public_prediction_response",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_linked_id_public_prediction_surfaces_match_mgcv_on_supported_gaussian_case",
        "response",
    ),
    _leaf(
        "general_stage_linked_id_public_prediction_terms",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_linked_id_public_prediction_surfaces_match_mgcv_on_supported_gaussian_case",
        "terms",
    ),
    _leaf(
        "general_stage_iterms_rejection_default",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_iterms_rejects_multi_predictor_public_surface",
        "default",
    ),
    _leaf(
        "general_stage_iterms_rejection_type_2",
        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        "test_general_family_iterms_rejects_multi_predictor_public_surface",
        "type_2",
    ),
)
