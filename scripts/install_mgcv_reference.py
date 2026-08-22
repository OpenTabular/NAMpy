#!/usr/bin/env python3
"""Install the tracked mgcv reference package into a repo-local R library."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstreams" / "mgcv"
DEFAULT_LIBRARY = ROOT / ".cache" / "mgcv-lib"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args()

    if not SOURCE.is_dir():
        parser.error(f"missing tracked mgcv source: {SOURCE}")
    r_command = shutil.which("R")
    if r_command is None:
        parser.error("R is required to install the mgcv reference package")

    library = args.library if args.library.is_absolute() else ROOT / args.library
    library.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(library)
    subprocess.run(
        [r_command, "CMD", "INSTALL", "-l", str(library), str(SOURCE)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(f"Installed mgcv reference into {library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
