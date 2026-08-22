#!/usr/bin/env python3
"""Fetch local upstream references from the tracked source catalogue."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "reference_generation" / "upstreams.json"
LOCKFILE = ROOT / "upstreams" / "lock.json"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def fetch(entry: dict, refresh: bool) -> dict:
    destination = ROOT / "upstreams" / entry["path"]
    if not (destination / ".git").exists():
        command = ["git", "clone", "--depth", "1"]
        if entry.get("sparse_paths"):
            command.append("--sparse")
        command.extend([entry["url"], str(destination)])
        subprocess.run(command, check=True)
    elif refresh:
        run("git", "fetch", "--depth", "1", "origin", cwd=destination)
        run("git", "reset", "--hard", "origin/HEAD", cwd=destination)

    sparse_paths = entry.get("sparse_paths")
    if sparse_paths:
        run("git", "sparse-checkout", "set", "--no-cone", *sparse_paths, cwd=destination)

    sha = run("git", "rev-parse", "HEAD", cwd=destination)
    return {"name": entry["name"], "path": entry["path"], "url": entry["url"], "sha": sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    lock = {"version": manifest["version"], "repositories": []}
    for entry in manifest["repositories"]:
        print(f"Fetching {entry['name']} -> upstreams/{entry['path']}")
        lock["repositories"].append(fetch(entry, args.refresh))
    LOCKFILE.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Wrote {LOCKFILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
