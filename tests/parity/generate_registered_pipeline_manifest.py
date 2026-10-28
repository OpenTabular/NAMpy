"""Record hashes and scope of the committed invariant reference captures.

After explicitly refreshing the relevant invariant suites, run with
NAMPY_REFRESH_REFERENCE_FIXTURES=1. Normal validation uses --check and needs no R.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tests.reference_fixtures import generation_enabled

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "tests/reference_fixtures/registered_seven_stage"
SOURCES = (
    "tests/parity/probes/mgcv_registered_pipeline.R",
    "tests/parity/_registered_pipeline_reference.py",
    "tests/parity/_seven_stage_cases.py",
    "tests/parity/_pipeline_invariants.py",
    "tests/parity/test_registered_smooth_seven_stage_pipeline.py",
    "tests/parity/test_recent_family_seven_stage_pipeline.py",
    "tests/parity/seven_stage_coverage.json",
    "tests/reference_fixtures/registered_seven_stage/smooth_responses.json",
)


def manifest(namespace="registered_seven_stage"):
    if namespace not in {
        "registered_seven_stage",
        "mgcv_smooth_family_crossings",
        "mgcv_k_check_shared_plan",
    }:
        raise ValueError(f"Unknown invariant reference namespace {namespace}")
    directory = DIRECTORY.parent / namespace
    expected_count = {
        "registered_seven_stage": 94,
        "mgcv_smooth_family_crossings": 72,
        "mgcv_k_check_shared_plan": 42,
    }[namespace]
    sources = SOURCES + (
        ("tests/parity/test_mgcv_smooth_family_crossings.py",)
        if namespace == "mgcv_smooth_family_crossings"
        else ()
    )
    if namespace == "mgcv_k_check_shared_plan":
        sources += (
            "tests/diagnostics/test_mgcv_k_check_shared_plan.py",
            "tests/parity/probes/mgcv_shared_plan_k_check.R",
            "tests/parity/test_mgcv_smooth_family_crossings.py",
        )
    captures = {}
    cases = set()
    versions = {}
    for path in sorted(directory.glob("*.json.gz")):
        payload = path.read_bytes()
        record = json.loads(gzip.decompress(payload))
        name = record["case_name"]
        if name in cases:
            raise ValueError(
                f"Duplicate/stale capture for {name}; remove superseded fixtures before recording."
            )
        cases.add(name)
        versions[record["engine"]] = record["version"]
        captures[path.name] = {
            "case": name,
            "engine": record["engine"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_warnings": record["source_warnings"],
            "upstream_notes": record["upstream_notes"],
        }
    if len(captures) != expected_count:
        raise ValueError(
            f"Expected {expected_count} independent captures, found {len(captures)}"
        )
    return {
        "schema_version": 1,
        "policy": "Normal tests read committed reference outputs only; no upstream execution. Comparisons use observable invariants, not basis columns, eigenvectors, coefficients or coefficient-coordinate covariance.",
        "upstream_packages": versions,
        "history_and_scope": "tests/parity/seven_stage_coverage.json",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        "capture_count": len(captures),
        "captures": captures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--namespace",
        choices=(
            "registered_seven_stage",
            "mgcv_smooth_family_crossings",
            "mgcv_k_check_shared_plan",
        ),
        default="registered_seven_stage",
    )
    args = parser.parse_args()
    path = DIRECTORY.parent / args.namespace / "manifest.json"
    actual = manifest(args.namespace)
    if args.check:
        assert actual == json.loads(path.read_text()), (
            "Invariant reference manifest is stale or damaged"
        )
        print(
            f"PASS {actual['capture_count']} upstream capture hashes and input/source provenance"
        )
    else:
        if not generation_enabled():
            raise RuntimeError(
                "Manifest generation requires explicit reference refresh mode"
            )
        path.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"Recorded {actual['capture_count']} upstream captures")


if __name__ == "__main__":
    main()
