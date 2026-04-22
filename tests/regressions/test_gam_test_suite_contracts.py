from __future__ import annotations

import ast
from pathlib import Path

from tests._taxonomy_registry import (
    _DEFAULT_MARKS_BY_FILE,
    _FAMILY_MARK_NAMES,
    _METHOD_MARK_NAMES,
    _PRIMARY_COVERAGE_BY_MARK,
    _SELECTION_CAPABLE_FILES,
    _SMOOTH_MARK_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"


def _iter_test_function_defs(tree: ast.AST):
    stack = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        for child in ast.iter_child_nodes(node):
            child_parents = parents + [node]
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, child_parents
            stack.append((child, child_parents))


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
