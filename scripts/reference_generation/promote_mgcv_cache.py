"""Promote legacy local mgcv JSON cache entries to static fixtures.

This is intentionally an explicit developer command. It does not execute
mgcv; it converts results already generated and reviewed locally.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    sources = sorted(args.source.glob("*.json"))
    if not sources:
        parser.error(f"no JSON cache entries found under {args.source}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        value = json.loads(source.read_text(encoding="utf-8"))
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        target = args.destination / f"{source.stem}.json.gz"
        target.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(f"Promoted {len(sources)} mgcv fixtures to {args.destination}")


if __name__ == "__main__":
    main()
