from __future__ import annotations

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
        "tests/regressions/test_gam_test_suite_contracts.py",
    ),
    "surface_backend": ("tests/optimization/test_mgcv_gaussian_backend_selection.py",),
}
