"""Check reference feature coverage, input contracts and executed example records."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))


def main():
    from neural_method_reference import METHODS
    from reference_schema import core_parameter_rows
    from reference_symbols import SYMBOLS

    from nampy.gam.model.api import _GAM_HPARAM_KEYS

    docs = ROOT / "docs"
    examples = docs / "_examples/reference"
    verification = json.loads((examples / "verification.json").read_text())
    schemas = json.loads((examples / "output_schemas.json").read_text())
    errors = []
    expected = set()
    for folder in ["statistical_reference", "neural_reference"]:
        for path in (docs / folder).rglob("*.rst"):
            # Index/architecture inventories organize features, rather than
            # defining a callable. Every remaining feature requires an example.
            text = path.read_text()
            if ".. toctree::" in text:
                if (
                    folder == "neural_reference"
                    and path.parent == docs / folder
                    and path.stem not in {"index", "internals"}
                ):
                    pass  # individual architecture pages also have task links
                else:
                    continue
            if path.name == "internals.rst":
                continue
            page = path.relative_to(docs).with_suffix("").as_posix()
            expected.add(page)
            if "_examples/reference/" + page + ".py" not in text:
                errors.append(f"{page}: missing runnable example include")
            if page not in {
                "statistical_reference/GAM",
                "statistical_reference/uncertainty",
                "statistical_reference/internals/covariance_correction",
            }:
                if (
                    "Type" not in text
                    and ".. include:: /_generated/reference/" not in text
                ):
                    errors.append(f"{page}: missing input type contract")
    for module, names in SYMBOLS.items():
        for name in names:
            expected.add(f"api/generated/{module}.{name}")
    expected.add("api/neural_training")
    for name in METHODS:
        expected.add("neural_reference/methods/" + name)
    for page in sorted(expected):
        path = examples / (page + ".py")
        if not path.exists():
            errors.append(f"{page}: missing standalone source")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = verification.get(page, {})
        if record.get("status") != "passed" or record.get("sha256") != digest:
            errors.append(
                f"{page}: example unexecuted, failed or changed since execution"
            )
        rows = schemas.get(page)
        if not rows or any(len(row) != 3 or not row[1] or not row[2] for row in rows):
            errors.append(f"{page}: missing verified output types/shapes")
    documented = {row[0].strip("`") for row in core_parameter_rows()}
    if documented != set(_GAM_HPARAM_KEYS):
        errors.append(
            "core GAM constructor keyword coverage differs from implementation"
        )
    method_index = (docs / "statistical_reference/methods/index.rst").read_text()
    if "Vb.corr" in method_index:
        errors.append("private Vb.corr is exposed as a standalone user method")
    if "Prediction uncertainty" not in method_index:
        errors.append("missing public uncertainty workflow")
    if errors:
        print("\n".join(errors))
        raise SystemExit(f"{len(errors)} reference contract failures")
    print(
        f"PASS: {len(expected)} feature examples match successful execution records; parameter types/defaults and output schemas are present."
    )


if __name__ == "__main__":
    main()
