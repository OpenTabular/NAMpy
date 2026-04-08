from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "parity_summary.md"


def _load_matrix_rows() -> list[dict[str, str]]:
    lines = MATRIX_PATH.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) != 9:
            continue
        if cols[0] == "metric":
            continue
        rows.append(
            {
                "metric": cols[0],
                "surface": cols[1],
                "family": cols[2],
                "formula_class": cols[3],
                "method": cols[4],
                "owner_test_file": cols[5].strip("`"),
                "owner_test_id": cols[6].strip("`"),
                "tolerance_policy": cols[7],
                "status": cols[8],
            }
        )
    return rows


def test_parity_matrix_has_no_duplicate_obligations():
    rows = _load_matrix_rows()
    assert rows, "No parity matrix rows found."
    seen: set[tuple[str, str, str, str, str]] = set()
    dupes: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        key = (
            row["metric"],
            row["surface"],
            row["family"],
            row["formula_class"],
            row["method"],
        )
        if key in seen:
            dupes.append(key)
        seen.add(key)
    assert not dupes, f"Duplicate parity obligations found: {dupes}"


def test_parity_matrix_owner_tests_exist():
    rows = _load_matrix_rows()
    missing_files: list[str] = []
    missing_tests: list[str] = []

    for row in rows:
        rel_file = row["owner_test_file"]
        test_id = row["owner_test_id"]
        file_path = REPO_ROOT / rel_file
        if not file_path.exists():
            missing_files.append(rel_file)
            continue
        text = file_path.read_text(encoding="utf-8")
        pattern = rf"^\s*def\s+{re.escape(test_id)}\s*\("
        if re.search(pattern, text, flags=re.MULTILINE) is None:
            missing_tests.append(f"{rel_file}:{test_id}")

    assert not missing_files, f"Missing owner test files: {missing_files}"
    assert not missing_tests, f"Missing owner tests: {missing_tests}"


def test_parity_matrix_status_and_tolerance_policy_are_valid():
    rows = _load_matrix_rows()
    valid_status = {"covered", "gap", "known_mismatch"}
    valid_tol = {"strict", "known_mismatch"}
    bad_status = [r for r in rows if r["status"] not in valid_status]
    bad_tol = [r for r in rows if r["tolerance_policy"] not in valid_tol]

    assert not bad_status, f"Invalid parity matrix status rows: {bad_status}"
    assert not bad_tol, f"Invalid parity matrix tolerance rows: {bad_tol}"
