from __future__ import annotations

import ast
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from tests._taxonomy_registry import (
    _DEFAULT_MARKS_BY_FILE,
    _DIRECT_SUPPORTED_LEAF_EXPECTATIONS,
    _FAMILY_MARK_NAMES,
    _METHOD_MARK_NAMES,
    _PRIMARY_COVERAGE_BY_MARK,
    _SELECTION_CAPABLE_FILES,
    _SMOOTH_MARK_NAMES,
    LeafCoverageExpectation,
)
from tests.families.test_general_family_mgcv_parity import GENERAL_SE_CASES
from tests.optimization._trace_parity_helpers import LINKED_ID_TRACE_CASES
from tests.parity.test_mgcv_general_family_prediction_stage_parity import (
    _BROADER_PREDICTION_STAGE_CASE_IDS,
    _METHOD_STAGE_CASES,
    _TENSOR_PUBLIC_GAP_CASES,
)
from tests.parity.test_mgcv_output_parity import (
    SE_SNAPSHOT_CASES,
    TERMS_PARITY_CASES,
    TRANSFORMED_SMOOTH_OUTPUT_CASES,
)
from tests.parity.test_mgcv_snapshot_core_matrix import CASES as SNAPSHOT_CORE_CASES

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
PYTEST_BIN = shutil.which("pytest") or "pytest"


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
    for param in _TENSOR_PUBLIC_GAP_CASES:
        for pred_type in ("link", "response", "terms"):
            expectations.append(
                LeafCoverageExpectation(
                    leaf_id=f"general_stage_tensor_gap_{param.id}_{pred_type}",
                    nodeid_parts=(
                        "tests/parity/test_mgcv_general_family_prediction_stage_parity.py",
                        "test_general_family_tensor_heavy_prediction_case_stays_localized_to_known_gap",
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
