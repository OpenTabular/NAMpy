"""Fail fast on repository states that make releases non-reproducible."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files(path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tracked_existing_files(path: str) -> list[str]:
    return [item for item in _tracked_files(path) if (ROOT / item).is_file()]


def main() -> None:
    errors: list[str] = []
    if (ROOT / "setup.py").exists():
        errors.append("setup.py duplicates the pyproject.toml package metadata")
    if tracked := _tracked_existing_files("tests/mgcv_r_cache"):
        errors.append(f"generated mgcv cache files are tracked: {tracked[:3]}")
    if tracked := _tracked_files("upstreams"):
        errors.append(f"local upstream sources are tracked: {tracked[:3]}")
    if tracked := _tracked_existing_files("docs/api/generated"):
        errors.append(f"generated autosummary files are tracked: {tracked[:3]}")

    if errors:
        raise SystemExit("Repository hygiene check failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
