#!/usr/bin/env python3
"""Verify upstream clone paths, remotes, and resolved commits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "reference_generation" / "upstreams.json"
LOCKFILE = ROOT / "upstreams" / "lock.json"


def git(destination: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=destination, text=True
    ).strip()


def normalize_remote(url: str) -> str:
    """Compare HTTPS remotes independent of an optional ``.git`` suffix."""
    return url.rstrip("/").removesuffix(".git")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    if not LOCKFILE.exists():
        print("Missing upstreams/lock.json; run fetch_upstreams.py first.", file=sys.stderr)
        return 1
    lock = {item["path"]: item for item in json.loads(LOCKFILE.read_text())["repositories"]}
    failures = []
    for entry in manifest["repositories"]:
        destination = ROOT / "upstreams" / entry["path"]
        item = lock.get(entry["path"])
        if item is None or not (destination / ".git").exists():
            failures.append(f"missing clone: {entry['name']}")
            continue
        actual_url = git(destination, "remote", "get-url", "origin")
        actual_sha = git(destination, "rev-parse", "HEAD")
        if normalize_remote(actual_url) != normalize_remote(entry["url"]):
            failures.append(f"remote mismatch: {entry['name']}: {actual_url}")
        if actual_sha != item["sha"]:
            failures.append(f"lock mismatch: {entry['name']}: {actual_sha}")
        print(f"OK {entry['name']} {actual_sha[:12]}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
