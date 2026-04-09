"""
Concatenate test files under ``tests/`` into one bundle.

Defaults:
- input root: tests/
- include pattern: test_*.py
- output file: combined_tests.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def collect_test_files(tests_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in tests_root.rglob("test_*.py"):
        if not path.is_file():
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(tests_root).as_posix())


def build_bundle(files: list[Path], tests_root: Path) -> str:
    lines: list[str] = []
    lines.append("Combined tests bundle\n")
    lines.append(f"Total files: {len(files)}\n")
    lines.append("\n")
    for path in files:
        rel = path.relative_to(tests_root).as_posix()
        lines.append(f"# --- Start of tests/{rel} ---\n")
        lines.append(path.read_text(encoding="utf-8"))
        lines.append(f"\n# --- End of tests/{rel} ---\n\n")
    return "".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Concatenate test files under tests/.")
    parser.add_argument(
        "--tests-root",
        default=str(repo_root / "tests"),
        help="Path to tests directory (default: ./tests).",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "combined_tests.txt"),
        help="Output file path (default: ./combined_tests.txt).",
    )
    args = parser.parse_args()

    tests_root = Path(args.tests_root).resolve()
    output_path = Path(args.output).resolve()

    files = collect_test_files(tests_root)
    bundle = build_bundle(files, tests_root)
    output_path.write_text(bundle, encoding="utf-8")

    print(f"Wrote {output_path} with {len(files)} files.")


if __name__ == "__main__":
    main()
