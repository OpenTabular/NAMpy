from __future__ import annotations

import ast
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from tests._taxonomy_registry import (
    _DEFAULT_MARKS_BY_FILE,
    _DIRECT_SUPPORTED_LEAF_EXPECTATIONS,
    _FAMILY_MARK_NAMES,
    _LINK_MARK_NAMES,
    _METHOD_MARK_NAMES,
    _OPTIMIZER_MARK_NAMES,
    _PRIMARY_COVERAGE_BY_MARK,
    _SELECTION_CAPABLE_FILES,
    _SMOOTH_MARK_NAMES,
    LeafCoverageExpectation,
)
from tests.conftest import _infer_marks_from_callspec
from tests.families.test_general_family_mgcv_parity import GENERAL_SE_CASES
from tests.optimization._trace_parity_helpers import LINKED_ID_TRACE_CASES
from tests.parity.test_mgcv_general_family_prediction_stage_parity import (
    _BROADER_PREDICTION_STAGE_CASE_IDS,
    _METHOD_STAGE_CASES,
)
from tests.parity.test_mgcv_output_parity import (
    SE_SNAPSHOT_CASES,
    TERMS_PARITY_CASES,
    TRANSFORMED_SMOOTH_OUTPUT_CASES,
)
from tests.parity.test_mgcv_snapshot_core_matrix import CASES as SNAPSHOT_CORE_CASES
from tests.parity.test_mgcv_under_tested_supported_combinations import (
    _GENERAL_STRUCTURED_CASES,
    _INFERENCE_FAMILIES,
    _NUMERICAL_STRESS_CASES,
    _OPTIMIZED_FORMULA_INTERSECTIONS,
    _OPTIMIZER_CRITERION_LINK_CASES,
    _STRUCTURED_TERMS,
    _TENSOR_BASES,
    _THREE_MARGIN_BASIS_COVER,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
PYTEST_BIN = shutil.which("pytest") or "pytest"
SUPPORTED_COMBINATIONS_PATH = (
    "tests/parity/test_mgcv_under_tested_supported_combinations.py"
)


def _surface_leaf(leaf_id: str, path: str, test_name: str, *parts: str):
    return LeafCoverageExpectation(
        leaf_id=leaf_id,
        nodeid_parts=(path, test_name, *parts),
    )


_GAM_PIPELINE_SURFACE_EXPECTATIONS = (
    _surface_leaf(
        "pipeline_1_formula_lists",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_1_formula_list_combines_shared_terms_transforms_offsets_and_intercepts",
    ),
    _surface_leaf(
        "pipeline_1_api_inputs",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_1_api_routes_weights_knots_min_sp_drop_intercept_and_fit_offset",
    ),
    _surface_leaf(
        "pipeline_1_supported_interactions_newdata",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_1_supported_numeric_factor_interactions_rebuild_on_newdata_like_mgcv",
    ),
    _surface_leaf(
        "pipeline_1_tensor_m_wrong_length_fallback",
        "tests/parity/test_gam_spec_build_owner_contracts.py",
        "test_tensor_m_wrong_length_warns_and_uses_mgcv_zero_fallback",
    ),
    *(
        _surface_leaf(
            f"pipeline_2_univariate_{basis}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_2_univariate_runtime_boundary_predictions_and_se_match_mgcv",
            basis,
        )
        for basis in ("cr", "cs", "cc", "ps", "tp", "ts")
    ),
    *(
        _surface_leaf(
            f"pipeline_2_structured_{basis}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_2_categorical_and_tensor_runtimes_pair_training_prediction_and_penalties",
            basis,
        )
        for basis in ("re", "fs", "sz", "te", "ti")
    ),
    *(
        _surface_leaf(
            f"pipeline_2_structured_boundary_{basis}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_2_structured_runtime_boundary_prediction_and_se_match_mgcv",
            basis,
        )
        for basis in ("re", "fs", "sz")
    ),
    *(
        _surface_leaf(
            f"pipeline_2_row_permutation_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_2_row_permutation_preserves_basis_space_penalties_and_behavior",
            case_id,
        )
        for case_id in (
            "linked_pooled",
            "fs_exchangeable_null_space",
            "ps",
            "tp_indeterminate_eigenspace",
            "ts_indeterminate_eigenspace",
            "tensor_te",
            "tensor_ti",
            "sz",
            "factor_by",
        )
    ),
    *(
        _surface_leaf(
            f"pipeline_3_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_3_wrapping_combinations_preserve_maps_and_penalty_ownership",
            case_id,
        )
        for case_id in (
            "numeric_by_linked",
            "factor_by_linked",
            "tensor_by",
            "mixed_select",
        )
    ),
    _surface_leaf(
        "pipeline_3_composed_coefficient_maps",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_3_composed_coefficient_maps_reproduce_each_constructed_basis",
    ),
    *(
        _surface_leaf(
            f"pipeline_3_fitted_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_3_wrapping_combinations_match_mgcv_fitted_behavior",
            case_id,
        )
        for case_id in (
            "numeric_by_linked",
            "factor_by_linked",
            "tensor_by",
            "mixed_select",
        )
    ),
    _surface_leaf(
        "pipeline_4_multi_predictor",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_4_multi_predictor_compilation_preserves_slices_offsets_and_intercepts",
    ),
    _surface_leaf(
        "pipeline_4_link_fixed_free_rank",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_4_three_term_linked_fixed_free_and_rank_deficient_assembly",
    ),
    *(
        _surface_leaf(
            f"pipeline_4_fitted_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_4_multi_predictor_fitted_assembly_matches_mgcv",
            case_id,
        )
        for case_id in (
            "gaulss_first_predictor_smooth",
            "gammals_second_predictor_smooth",
        )
    ),
    *(
        _surface_leaf(
            f"pipeline_5_{case_id}",
            "tests/optimization/test_mgcv_gam_side_parity.py",
            test_name,
            *parts,
        )
        for case_id, test_name, parts in (
            (
                "repeated_one_dimensional",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("repeat_cr_ps",),
            ),
            (
                "nested_main_effect",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("nested_tp",),
            ),
            (
                "nested_reverse_formula_order",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("nested_tp_reversed_formula_order",),
            ),
            (
                "nested_tensor",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("nested_te",),
            ),
            (
                "tensor_interaction_main_effects",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("main_ti",),
            ),
            (
                "exempt_plus_nested",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("re_plus_nested_tp",),
            ),
            (
                "three_way_nested",
                "test_three_way_nested_side_conditions_match_mgcv_behavior_invariantly",
                (),
            ),
            (
                "near_rank_boundary",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("near_rank_boundary_cr",),
            ),
            (
                "zero_width",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("zero_width_duplicate_cr",),
            ),
            (
                "ordered_factor_by",
                "test_ordered_factor_by_keeps_nonreference_levels_with_mgcv_blocks",
                (),
            ),
            (
                "no_intercept",
                "test_gam_side_matches_mgcv_nested_side_condition_cases",
                ("gaussian_no_intercept_nested_tp",),
            ),
            (
                "general_family",
                "test_gam_side_matches_mgcv_current_general_case_matrix",
                ("gaulss_cr",),
            ),
            (
                "sign_indeterminate_sz",
                "test_gam_side_matches_mgcv_current_non_general_case_matrix",
                ("gaussian_sz",),
            ),
        )
    ),
    *(
        _surface_leaf(
            f"pipeline_5_rank_neighborhood_{case_id}",
            "tests/optimization/test_mgcv_gam_side_parity.py",
            "test_gam_side_rank_threshold_neighborhood_matches_mgcv",
            case_id,
        )
        for case_id in ("effectively_singular", "identified_two_dimensional")
    ),
    _surface_leaf(
        "pipeline_5_multi_predictor",
        "tests/optimization/test_mgcv_gam_side_parity.py",
        "test_gam_side_multi_predictor_blocks_match_mgcv_independently",
    ),
    *(
        _surface_leaf(
            f"pipeline_6_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_6_combined_family_link_criterion_optimizer_fit_matches_mgcv",
            case_id,
        )
        for case_id in (
            "gaussian_gcv_weight_offset_select",
            "binomial_probit_reml_bfgs_weight_offset_select",
            "gamma_log_ml_bfgs_zero_weight",
            "negbin_est_reml_optim_weight_offset",
            "poisson_reml_newton_weight_offset",
            "gaulss_ml_newton",
            "gammals_ml_newton",
        )
    ),
    _surface_leaf(
        "pipeline_6_near_separated_binomial",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_6_near_separated_binomial_fixed_fit_matches_mgcv",
    ),
    *(
        _surface_leaf(
            f"pipeline_6_zero_weight_{family}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_6_all_zero_prior_weights_fail_explicitly",
            family,
        )
        for family in ("gaussian", "poisson", "gaulss")
    ),
    _surface_leaf(
        "pipeline_7_filters_unconditional",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_7_terms_filter_unconditional_se_matches_mgcv",
    ),
    *(
        _surface_leaf(
            f"pipeline_7_structured_unconditional_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_7_structured_terms_unconditional_se_matches_mgcv",
            case_id,
        )
        for case_id in (
            "random_effect",
            "tensor",
            "linked_id",
            "fs",
            "sz",
            "factor_by",
            "ti",
        )
    ),
    *(
        _surface_leaf(
            f"pipeline_7_newdata_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_7_structured_term_newdata_response_matches_mgcv",
            case_id,
        )
        for case_id in ("random_effect", "factor_by", "tensor", "linked_id")
    ),
    _surface_leaf(
        "pipeline_7_rejected_newdata",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_7_prediction_rejects_unseen_factor_levels_and_nan_explicitly",
    ),
    _surface_leaf(
        "pipeline_7_diagnostics_persistence",
        "tests/parity/test_gam_pipeline_combination_matrix.py",
        "test_stage_7_fit_diagnostics_snapshot_and_persistence_roundtrip",
    ),
    _surface_leaf(
        "pipeline_7_gaussian_inverse_postfit",
        "tests/parity/test_mgcv_under_tested_supported_combinations.py",
        "test_gaussian_inverse_reml_postfit_surfaces_match_mgcv",
    ),
    *(
        _surface_leaf(
            f"pipeline_7_persistence_{case_id}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_7_structured_and_general_persistence_roundtrip",
            case_id,
        )
        for case_id in (
            "random_effect",
            "tensor",
            "linked_id",
            "gaulss",
            "fs",
            "sz",
            "ti",
            "factor_by",
            "gammals",
            "binomial_pirls",
            "poisson_pirls",
            "gamma_pirls",
        )
    ),
    *(
        _surface_leaf(
            f"pipeline_7_three_model_anova_{family}",
            "tests/parity/test_gam_pipeline_combination_matrix.py",
            "test_stage_7_three_model_rank_deficient_anova_matches_mgcv",
            family,
        )
        for family in (
            "gaussian",
            "binomial",
            "poisson",
            "gamma",
            "gaulss",
            "gammals",
        )
    ),
    _surface_leaf(
        "formula_parser_single",
        "tests/parity/test_mgcv_formula_parse_parity.py",
        "test_single_formula_parser_matches_mgcv_interpret_gam",
    ),
    _surface_leaf(
        "formula_parser_list",
        "tests/parity/test_mgcv_formula_parse_parity.py",
        "test_formula_list_parser_matches_mgcv_interpret_gam",
    ),
    _surface_leaf(
        "formula_preprocess_transforms",
        "tests/parity/test_mgcv_formula_parse_parity.py",
        "test_apply_formula_preprocess_rebuilds_transformed_smooth_covariates",
    ),
    _surface_leaf(
        "prediction_all_types_and_filters",
        "tests/diagnostics/test_gam_plot_and_public_api_contracts.py",
        "test_predict_terms_and_exclude_filters_match_mgcv",
    ),
    _surface_leaf(
        "mixed_fixed_free_sp",
        "tests/diagnostics/test_gam_plot_and_public_api_contracts.py",
        "test_mixed_fixed_and_free_smoothing_parameters_fit_and_predict",
    ),
    _surface_leaf(
        "public_gam_exports",
        "tests/diagnostics/test_gam_plot_and_public_api_contracts.py",
        "test_public_gam_package_exports_contract",
    ),
    *(
        _surface_leaf(
            f"unsupported_{guard}",
            "tests/optimization/test_gam_unsupported_branch_guards.py",
            test_name,
        )
        for guard, test_name in (
            ("formula_function", "test_formula_unsupported_expression_function_guard_raises_explicitly"),
            ("formula_list_dot", "test_formula_list_data_aware_dot_shorthand_guard_raises_explicitly"),
            ("fs_shrinkage", "test_fs_full_rank_shrinkage_surface_guard_raises_explicitly"),
            ("re_linked_id", "test_random_effect_linked_id_guard_raises_explicitly"),
            ("negbin_ml_optim", "test_negbin_estimated_theta_ml_optim_guard_raises_explicitly"),
            ("t2", "test_t2_smooth_guard_raises_explicitly"),
            ("point_constraints", "test_pc_guard_raises_explicitly_outside_supported_univariate_bases"),
            ("unknown_constructor", "test_gam_unknown_constructor_arguments_raise_explicitly"),
            ("gacv", "test_gacv_cp_method_guard_raises_explicitly"),
            ("general_term_filters", "test_general_family_prediction_term_filter_guard_raises_explicitly"),
            ("shared_lpi_component", "test_shared_lpi_component_guard_raises_explicitly"),
        )
    ),
)


def _iter_test_function_defs(tree: ast.AST):
    stack = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        for child in ast.iter_child_nodes(node):
            child_parents = parents + [node]
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, child_parents
            stack.append((child, child_parents))


@lru_cache(maxsize=None)
def _collected_nodeids_for_paths(*rel_paths: str) -> tuple[str, ...]:
    result = subprocess.run(
        [PYTEST_BIN, "--collect-only", "-qq", *rel_paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    )


def _missing_leaf_ids(
    nodeids: list[str], expectations: tuple[LeafCoverageExpectation, ...]
) -> list[str]:
    missing: list[str] = []
    for expectation in expectations:
        if not any(
            all(part in nodeid for part in expectation.nodeid_parts)
            for nodeid in nodeids
        ):
            missing.append(expectation.leaf_id)
    return missing


def _snapshot_core_leaf_expectations() -> tuple[LeafCoverageExpectation, ...]:
    return tuple(
        LeafCoverageExpectation(
            leaf_id=f"snapshot_core_{case.case_id}",
            nodeid_parts=(
                "tests/parity/test_mgcv_snapshot_core_matrix.py",
                "test_requested_mgcv_parity_models",
                case.case_id,
            ),
        )
        for case in SNAPSHOT_CORE_CASES
    )


def _general_family_leaf_expectations() -> tuple[LeafCoverageExpectation, ...]:
    expectations: list[LeafCoverageExpectation] = []
    for case_id, *_rest in GENERAL_SE_CASES:
        expectations.extend(
            [
                LeafCoverageExpectation(
                    leaf_id=f"general_family_outer_derivatives_{case_id}",
                    nodeid_parts=(
                        "tests/families/test_general_family_mgcv_parity.py",
                        "test_general_family_fixed_sp_outer_derivatives_match_mgcv_across_surface",
                        case_id,
                    ),
                ),
                LeafCoverageExpectation(
                    leaf_id=f"general_family_outer_fit_{case_id}",
                    nodeid_parts=(
                        "tests/families/test_general_family_mgcv_parity.py",
                        "test_general_family_outer_fit_matches_mgcv_endpoint_across_surface",
                        case_id,
                    ),
                ),
                LeafCoverageExpectation(
                    leaf_id=f"general_family_snapshot_standard_errors_{case_id}",
                    nodeid_parts=(
                        "tests/families/test_general_family_mgcv_parity.py",
                        "test_general_family_link_response_standard_errors_match_mgcv_snapshot",
                        case_id,
                    ),
                ),
            ]
        )
        for pred_type in ("link", "response", "terms", "lpmatrix"):
            expectations.append(
                LeafCoverageExpectation(
                    leaf_id=f"general_family_newdata_{case_id}_{pred_type}",
                    nodeid_parts=(
                        "tests/families/test_general_family_mgcv_parity.py",
                        "test_general_family_newdata_prediction_surfaces_match_mgcv",
                        case_id,
                        pred_type,
                    ),
                )
            )
        for pred_type in ("link", "response", "terms"):
            expectations.append(
                LeafCoverageExpectation(
                    leaf_id=f"general_family_unconditional_{case_id}_{pred_type}",
                    nodeid_parts=(
                        "tests/families/test_general_family_mgcv_parity.py",
                        "test_general_family_newdata_unconditional_standard_errors_match_mgcv",
                        case_id,
                        pred_type,
                    ),
                )
            )
    return tuple(expectations)


def _output_leaf_expectations() -> tuple[LeafCoverageExpectation, ...]:
    expectations: list[LeafCoverageExpectation] = []
    for case in TERMS_PARITY_CASES:
        case_id = case["case_id"]
        for se_id in ("no_se", "with_se"):
            expectations.append(
                LeafCoverageExpectation(
                    leaf_id=f"output_terms_{case_id}_{se_id}",
                    nodeid_parts=(
                        "tests/parity/test_mgcv_output_parity.py",
                        "test_output_parity_terms",
                        case_id,
                        se_id,
                    ),
                )
            )
    for case in TRANSFORMED_SMOOTH_OUTPUT_CASES:
        case_id = case["case_id"]
        expectations.extend(
            [
                LeafCoverageExpectation(
                    leaf_id=f"output_transformed_smooth_terms_{case_id}",
                    nodeid_parts=(
                        "tests/parity/test_mgcv_output_parity.py",
                        "test_output_parity_newdata_transformed_smooth_terms",
                        case_id,
                    ),
                ),
                LeafCoverageExpectation(
                    leaf_id=f"output_transformed_smooth_lpmatrix_{case_id}",
                    nodeid_parts=(
                        "tests/parity/test_mgcv_output_parity.py",
                        "test_output_parity_newdata_transformed_smooth_lpmatrix",
                        case_id,
                    ),
                ),
            ]
        )
    for case_id, *_rest in SE_SNAPSHOT_CASES:
        expectations.append(
            LeafCoverageExpectation(
                leaf_id=f"output_snapshot_standard_errors_{case_id}",
                nodeid_parts=(
                    "tests/parity/test_mgcv_output_parity.py",
                    "test_output_parity_snapshot_link_and_response_standard_errors",
                    case_id,
                ),
            )
        )
    return tuple(expectations)


def _trace_leaf_expectations() -> tuple[LeafCoverageExpectation, ...]:
    expectations: list[LeafCoverageExpectation] = []
    for param in LINKED_ID_TRACE_CASES:
        expectations.append(
            LeafCoverageExpectation(
                leaf_id=f"trace_{param.id}",
                nodeid_parts=(
                    "tests/smooths/test_mgcv_linked_id_trace_parity.py",
                    "test_gaussian_linked_id_reml_score_hist_matches_mgcv_supported_bases",
                    param.id,
                ),
            )
        )
    return tuple(expectations)


def _general_stage_leaf_expectations() -> tuple[LeafCoverageExpectation, ...]:
    expectations: list[LeafCoverageExpectation] = []
    for case_id in _BROADER_PREDICTION_STAGE_CASE_IDS:
        for pred_type in ("link", "response", "terms"):
            expectations.extend(
                [
                    LeafCoverageExpectation(
                        leaf_id=f"general_stage_matrix_prediction_{case_id}_{pred_type}",
                        nodeid_parts=(
                            "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
                            "test_general_family_stage_matrix_newdata_prediction_surfaces_match_mgcv",
                            case_id,
                            pred_type,
                        ),
                    ),
                    LeafCoverageExpectation(
                        leaf_id=(
                            f"general_stage_matrix_unconditional_{case_id}_{pred_type}"
                        ),
                        nodeid_parts=(
                            "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
                            "test_general_family_stage_matrix_unconditional_standard_errors_match_mgcv",
                            case_id,
                            pred_type,
                        ),
                    ),
                ]
            )
    for param in _METHOD_STAGE_CASES:
        for pred_type in ("link", "response", "terms"):
            expectations.append(
                LeafCoverageExpectation(
                    leaf_id=f"general_stage_method_{param.id}_{pred_type}",
                    nodeid_parts=(
                        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
                        "test_general_family_method_stage_prediction_surfaces_match_mgcv",
                        param.id,
                        pred_type,
                    ),
                )
            )
    return tuple(expectations)


def _direct_leaf_expectations_for_prefixes(
    *prefixes: str,
) -> tuple[LeafCoverageExpectation, ...]:
    return tuple(
        expectation
        for expectation in _DIRECT_SUPPORTED_LEAF_EXPECTATIONS
        if expectation.nodeid_parts and expectation.nodeid_parts[0] in prefixes
    )


def test_primary_coverage_catalog_covers_declared_taxonomy_marks():
    """Verify that primary coverage catalog covers declared taxonomy marks."""
    required_marks = {
        *_SMOOTH_MARK_NAMES.values(),
        *_FAMILY_MARK_NAMES.values(),
        *_METHOD_MARK_NAMES.values(),
        *_LINK_MARK_NAMES.values(),
        *_OPTIMIZER_MARK_NAMES.values(),
        "select_true",
        "select_false",
        "surface_snapshot",
        "surface_output",
        "surface_smoothcon",
        "surface_trace",
        "surface_kcheck",
        "surface_derivatives",
        "surface_regression",
        "surface_backend",
    }
    assert required_marks <= set(_PRIMARY_COVERAGE_BY_MARK)


def test_primary_coverage_catalog_paths_exist():
    """Verify that primary coverage catalog paths exist."""
    for mark_name, rel_paths in sorted(_PRIMARY_COVERAGE_BY_MARK.items()):
        assert rel_paths, f"{mark_name} missing primary coverage paths"
        for rel_path in rel_paths:
            assert (
                REPO_ROOT / rel_path
            ).exists(), f"{mark_name} references missing file {rel_path}"


def test_selection_capable_files_have_default_surface_marks():
    """Verify that selection-capable files have default surface marks."""
    for filename in sorted(_SELECTION_CAPABLE_FILES):
        assert (
            filename in _DEFAULT_MARKS_BY_FILE
        ), f"{filename} missing default surface mark entry"


def test_pytest_collection_does_not_hide_nested_test_functions():
    """Verify that pytest collection does not hide nested test functions."""
    nested = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, parents in _iter_test_function_defs(tree):
            if not node.name.startswith("test_"):
                continue
            fn_parents = [
                parent
                for parent in parents
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if fn_parents:
                nested.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.name}"
                )
    assert nested == []


def test_direct_supported_leaf_registry_has_unique_ids():
    """Verify that direct supported leaf registry uses unique leaf ids."""
    leaf_ids = [
        expectation.leaf_id for expectation in _DIRECT_SUPPORTED_LEAF_EXPECTATIONS
    ]
    assert len(leaf_ids) == len(set(leaf_ids))


def test_snapshot_leaf_paths_have_collected_owner_tests():
    """Verify that snapshot leaf paths have collected owner tests."""
    nodeids = list(
        _collected_nodeids_for_paths(
            "tests/parity/test_mgcv_snapshot_core_matrix.py",
            "tests/parity/test_mgcv_snapshot_extended_matrix.py",
            "tests/smooths/test_mgcv_pc_id_parity.py",
        )
    )
    missing = _missing_leaf_ids(
        nodeids,
        _snapshot_core_leaf_expectations()
        + _direct_leaf_expectations_for_prefixes(
            "tests/parity/test_mgcv_snapshot_extended_matrix.py",
            "tests/smooths/test_mgcv_pc_id_parity.py",
        ),
    )
    assert missing == []


def test_general_family_leaf_paths_have_collected_owner_tests():
    """Verify that general-family leaf paths have collected owner tests."""
    nodeids = list(
        _collected_nodeids_for_paths(
            "tests/families/test_general_family_mgcv_parity.py",
            "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        )
    )
    missing = _missing_leaf_ids(
        nodeids,
        _general_family_leaf_expectations()
        + _general_stage_leaf_expectations()
        + _direct_leaf_expectations_for_prefixes(
            "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
        ),
    )
    assert missing == []


def test_output_leaf_paths_have_collected_owner_tests():
    """Verify that output leaf paths have collected owner tests."""
    nodeids = list(
        _collected_nodeids_for_paths(
            "tests/parity/test_mgcv_output_parity.py",
        )
    )
    missing = _missing_leaf_ids(
        nodeids,
        _output_leaf_expectations(),
    )
    missing.extend(
        _missing_leaf_ids(
            nodeids,
            _direct_leaf_expectations_for_prefixes(
                "tests/parity/test_mgcv_output_parity.py",
            ),
        )
    )
    assert missing == []


def test_trace_leaf_paths_have_collected_owner_tests():
    """Verify that trace leaf paths have collected owner tests."""
    nodeids = list(
        _collected_nodeids_for_paths(
            "tests/smooths/test_mgcv_linked_id_trace_parity.py",
        )
    )
    missing = _missing_leaf_ids(
        nodeids,
        _trace_leaf_expectations(),
    )
    assert missing == []


def test_all_registered_supported_leaves_have_collected_owner_tests():
    """Every declared supported leaf remains backed by a collected behavioral test."""
    paths = sorted(
        {
            expectation.nodeid_parts[0]
            for expectation in _DIRECT_SUPPORTED_LEAF_EXPECTATIONS
        }
    )
    nodeids = list(_collected_nodeids_for_paths(*paths))
    assert _missing_leaf_ids(nodeids, _DIRECT_SUPPORTED_LEAF_EXPECTATIONS) == []


def test_every_gam_pipeline_stage_and_guard_has_a_collected_owner_test():
    """The seven-stage integration matrix and unsupported guards are enforceable."""
    paths = sorted(
        {
            expectation.nodeid_parts[0]
            for expectation in _GAM_PIPELINE_SURFACE_EXPECTATIONS
        }
    )
    nodeids = list(_collected_nodeids_for_paths(*paths))
    assert _missing_leaf_ids(nodeids, _GAM_PIPELINE_SURFACE_EXPECTATIONS) == []


def test_supported_combination_registries_retain_declared_axes_and_cases():
    """The broad fitted matrix must not silently lose a supported axis or case."""
    assert tuple(_TENSOR_BASES) == ("cr", "cs", "cc", "ps", "tp", "ts")
    assert tuple(_THREE_MARGIN_BASIS_COVER) == (
        ("cr", "ps", "tp"),
        ("cs", "tp", "cc"),
        ("cc", "ts", "ps"),
        ("ps", "cr", "ts"),
        ("tp", "cs", "cr"),
        ("ts", "cc", "cs"),
    )
    assert {case.term_id for case in _STRUCTURED_TERMS} == {
        "re",
        "fs",
        "sz",
        "te",
        "ti",
    }
    assert {case.family_id for case in _INFERENCE_FAMILIES} == {
        "gaussian",
        "binomial",
        "poisson",
        "gamma",
        "negbin",
    }
    assert {case.case_id for case in _OPTIMIZER_CRITERION_LINK_CASES} == {
        "gaussian_log_ml_bfgs_select_mixed_ti",
        "poisson_sqrt_reml_efs_mixed_te",
        "binomial_cauchit_ml_optim_fs",
        "gaussian_identity_gcv_newton_weighted_offset_re",
        "binomial_probit_reml_bfgs_select_weighted_offset_sz",
        "gamma_inverse_ml_optim_weighted_offset_te",
        "negbin_sqrt_reml_efs_weighted_offset_re",
        "poisson_log_reml_optim_select_weighted_offset_fs",
        "gaussian_reml_newton_boundary_re",
    }
    assert {case.case_id for case in _NUMERICAL_STRESS_CASES} == {
        "large_n_large_k",
        "severe_knot_clustering",
        "extreme_poisson_response_weights",
        "sparse_fs_cells",
        "sparse_sz_cells",
        "near_rank_seed_276",
        "near_rank_seed_277",
        "near_rank_seed_278",
    }
    assert {case.case_id for case in _OPTIMIZED_FORMULA_INTERSECTIONS} == {
        "binomial_cloglog_linked_factor_by_tensor_select_weights",
        "gamma_log_linked_factor_by_tensor_ml_weights",
        "gaulss_formula_list_tensor_offsets_select_weights",
        "gammals_formula_list_transform_by_offsets_weights",
    }
    assert {case.case_id for case in _GENERAL_STRUCTURED_CASES} == {
        "gaulss_re_fixed",
        "gaulss_fs_fixed",
        "gammals_sz_fixed",
        "gaulss_fs_second_predictor_fixed",
    }


def test_supported_combination_registries_are_fully_collected():
    """Every declared matrix entry must survive pytest parameter collection."""
    nodeids = _collected_nodeids_for_paths(SUPPORTED_COMBINATIONS_PATH)

    def selected(test_name: str) -> list[str]:
        return [nodeid for nodeid in nodeids if f"::{test_name}[" in nodeid]

    assert len(
        selected(
            "test_tensor_portable_cartesian_basis_matrix_matches_mgcv_response_and_se"
        )
    ) == 50
    assert (
        f"{SUPPORTED_COMBINATIONS_PATH}::"
        "test_tensor_cs_pairs_are_owned_by_representation_invariant_stage_checks"
        in nodeids
    )
    assert len(
        selected("test_three_margin_tensor_basis_cover_fixed_fit_matches_live_mgcv")
    ) == 12
    assert len(
        selected("test_structured_term_family_inference_cartesian_matrix_matches_mgcv")
    ) == 25

    for test_name, cases in (
        (
            "test_optimizer_criterion_link_structured_crosses_match_mgcv",
            _OPTIMIZER_CRITERION_LINK_CASES,
        ),
        (
            "test_numerical_stress_matrix_matches_live_mgcv_in_identified_space",
            _NUMERICAL_STRESS_CASES,
        ),
        (
            "test_optimized_formula_feature_intersections_match_live_mgcv",
            _OPTIMIZED_FORMULA_INTERSECTIONS,
        ),
        (
            "test_single_structured_term_general_family_fit_and_newdata_match_mgcv",
            _GENERAL_STRUCTURED_CASES,
        ),
    ):
        collected = selected(test_name)
        assert len(collected) == len(cases)
        assert all(
            any(case.case_id in nodeid for nodeid in collected) for case in cases
        )


def test_supported_combination_taxonomy_marks_select_representative_cases():
    """Link, criterion and optimizer marks remain wired to real matrix leaves."""
    expected_by_case = {
        "gaussian_log_ml_bfgs_select_mixed_ti": {
            "link_log",
            "method_ml",
            "optimizer_bfgs",
        },
        "poisson_sqrt_reml_efs_mixed_te": {
            "link_sqrt",
            "method_reml",
            "optimizer_efs",
        },
        "binomial_cauchit_ml_optim_fs": {
            "link_cauchit",
            "method_ml",
            "optimizer_optim",
        },
        "gaussian_identity_gcv_newton_weighted_offset_re": {
            "link_identity",
            "method_gcv",
            "optimizer_newton",
        },
        "binomial_probit_reml_bfgs_select_weighted_offset_sz": {
            "link_probit",
            "method_reml",
            "optimizer_bfgs",
        },
        "gamma_inverse_ml_optim_weighted_offset_te": {
            "link_inverse",
            "method_ml",
            "optimizer_optim",
        },
        "negbin_sqrt_reml_efs_weighted_offset_re": {
            "link_sqrt",
            "method_reml",
            "optimizer_efs",
        },
        "poisson_log_reml_optim_select_weighted_offset_fs": {
            "link_log",
            "method_reml",
            "optimizer_optim",
        },
        "gaussian_reml_newton_boundary_re": {
            "link_identity",
            "method_reml",
            "optimizer_newton",
        },
    }
    for case in _OPTIMIZER_CRITERION_LINK_CASES:
        item = SimpleNamespace(callspec=SimpleNamespace(params={"case": case}))
        marks = _infer_marks_from_callspec(item)
        assert expected_by_case[case.case_id] <= marks

    formula_case = next(
        case
        for case in _OPTIMIZED_FORMULA_INTERSECTIONS
        if case.case_id.startswith("binomial_cloglog")
    )
    formula_marks = _infer_marks_from_callspec(
        SimpleNamespace(callspec=SimpleNamespace(params={"case": formula_case}))
    )
    assert "link_cloglog" in formula_marks

    binomial_case = next(
        case for case in _INFERENCE_FAMILIES if case.family_id == "binomial"
    )
    default_link_marks = _infer_marks_from_callspec(
        SimpleNamespace(
            callspec=SimpleNamespace(params={"family_case": binomial_case})
        )
    )
    assert "link_logit" in default_link_marks

    ubre_marks = _infer_marks_from_callspec(
        SimpleNamespace(callspec=SimpleNamespace(params={"method": "UBRE"}))
    )
    assert "method_ubre" in ubre_marks


def test_every_test_module_is_collected_by_directory_collection():
    """Directory-mode collection must reach every test module on disk.

    The other collection contracts pass explicit file paths to pytest, which
    bypasses the ``python_files`` patterns in pyproject.toml — so they would
    keep certifying a file that a real ``pytest tests`` run silently skips
    (this happened: a name-prefix allowlist dropped five parity files).
    This guard collects by directory, exactly like a real run.
    """
    nodeids = _collected_nodeids_for_paths("tests")
    collected_files = {nodeid.split("::", 1)[0] for nodeid in nodeids}
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(TESTS_ROOT.rglob("test_*.py"))
        if str(path.relative_to(REPO_ROOT)) not in collected_files
    ]
    assert missing == []


def test_shared_snapshot_parity_classes_are_all_collected():
    """Every shared snapshot-parity test method must have a collected owner.

    ``tests/_mgcv_snapshot_parity_shared.py`` is uncollectable itself (leading
    underscore), so its classes only run when a collected module re-exports
    them. A refactor once dropped 54 methods this way; this guard fails when
    any shared ``Test*`` method has no collected counterpart under
    ``tests/parity``.
    """
    shared_path = TESTS_ROOT / "_mgcv_snapshot_parity_shared.py"
    tree = ast.parse(shared_path.read_text(encoding="utf-8"))
    shared_methods: list[tuple[str, str]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and child.name.startswith("test_"):
                    shared_methods.append((node.name, child.name))
    assert shared_methods, "shared snapshot module lost its test classes"

    nodeids = _collected_nodeids_for_paths("tests/parity")
    collected_methods = {
        nodeid.rsplit("::", 1)[-1].split("[", 1)[0] for nodeid in nodeids
    }
    missing = [
        f"{class_name}::{method_name}"
        for class_name, method_name in shared_methods
        if method_name not in collected_methods
    ]
    assert missing == []
