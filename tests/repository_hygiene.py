"""Fail fast on repository states that make releases non-reproducible."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL_ONLY_ROOT_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "GAM_IMPLEMENTED.md",
    "GAM_NOT_IMPLEMENTED.md",
    "PROJECT_STATUS.md",
    "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md",
    "UPSTREAM_LEDGER.md",
    "backlog.md",
    "paper_list.md",
}


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
    if (ROOT / "tests" / "mgcv_r_cache").exists():
        errors.append("obsolete tests/mgcv_r_cache directory exists")
    if tracked := _tracked_files("upstreams"):
        errors.append(f"local upstream sources are tracked: {tracked[:3]}")
    tracked_markdown = _tracked_files("*.md")
    tracked_local_notes = sorted(
        path
        for path in tracked_markdown
        if path in LOCAL_ONLY_ROOT_FILES
        or ("/" not in path and "review" in path.lower())
    )
    if tracked_local_notes:
        errors.append(f"local review/status notes are tracked: {tracked_local_notes[:3]}")
    if tracked := _tracked_files("scripts"):
        errors.append(f"local development scripts are tracked: {tracked[:3]}")
    if tracked := _tracked_files("debug"):
        errors.append(f"local debug probes are tracked: {tracked[:3]}")
    if tracked := _tracked_existing_files("docs/api/generated"):
        errors.append(f"generated autosummary files are tracked: {tracked[:3]}")

    if errors:
        raise SystemExit("Repository hygiene check failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
