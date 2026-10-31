"""Check reference feature coverage, input contracts and executed example records."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/scripts"))
sys.path.insert(0, str(ROOT / "docs/_ext"))


def main():
    from check_reference_examples import implementation_fingerprint
    from neural_method_reference import METHODS, OPTIONAL_HOOKS, supported_architectures
    from reference_schema import core_parameter_rows
    from reference_symbols import SYMBOLS

    from nampy.gam.model.api import _GAM_HPARAM_KEYS

    docs = ROOT / "docs"
    # CI runs source-contract tests before Sphinx creates include fragments.
    if not (docs / "_generated/reference/NAMRegressor.inc").exists():
        from public_reference import generate

        generate(ROOT)
    if any(
        not (docs / f"api/generated/{module}.{name}.rst").exists()
        for module, names in SYMBOLS.items()
        for name in names
    ):
        from reference_symbols import generate

        generate(docs)
    examples = docs / "_examples/reference"
    verification = json.loads((examples / "verification.json").read_text())
    implementation = implementation_fingerprint()
    schemas = json.loads((examples / "output_schemas.json").read_text())
    errors = []
    expected = set()
    inventory = json.loads((docs / "_data/public_reference_inventory.json").read_text())
    from public_reference import constructor_defaults, literal

    from nampy import models
    from nampy.gam.basis_registry import basis_descriptors
    from nampy.neural.families.registry import FAMILY_REGISTRY
    from nampy.neural.registry import architectures

    config_classes = {
        spec.config.__name__: spec.config for spec in architectures().values()
    }
    for feature in inventory:
        page = feature["page"]
        target = docs / (page + ".rst")
        if not target.exists():
            errors.append(f"{page}: missing canonical page")
            continue
        text = target.read_text()
        if feature.get("anchor") and f".. _{feature['anchor']}:" not in text:
            errors.append(f"{page}: missing canonical feature anchor")
        expected.add(feature["example"])
        for test in feature["tests"]:
            if not (ROOT / test).exists():
                errors.append(f"{page}: missing test {test}")
        if feature["kind"] == "architecture":
            from dataclasses import fields

            config = config_classes[feature["symbol"]]()
            actual = {f.name: literal(getattr(config, f.name)) for f in fields(config)}
            if actual != {p["name"]: p["default"] for p in feature["parameters"]}:
                errors.append(
                    f"{page}: configuration factory defaults differ from inventory"
                )
        if (
            feature.get("runtime_page")
            and not (docs / (feature["runtime_page"] + ".rst")).exists()
        ):
            errors.append(f"{page}: missing runtime constructor contract")
        if feature["kind"] == "estimator":
            cls = getattr(models, feature["symbol"])
            actual = {n: literal(v) for n, v in constructor_defaults(cls).items()}
            recorded = {p["name"]: p["default"] for p in feature["parameters"]}
            if actual != recorded:
                errors.append(
                    f"{page}: inventory defaults differ from effective constructor"
                )
            include = docs / f"_generated/reference/{feature['symbol']}.inc"
            contract = include.read_text() if include.exists() else text
            for n in actual:
                if f"``{n}``" not in contract:
                    errors.append(f"{page}: undocumented parameter {n}")
            if "Configuration option of type" in contract:
                errors.append(f"{page}: placeholder parameter meaning")
    for kind, symbols in [
        ("smooth", {d.name for d in basis_descriptors()}),
        ("neural_family", set(FAMILY_REGISTRY)),
    ]:
        if {f["symbol"] for f in inventory if f["kind"] == kind} != symbols:
            errors.append(f"{kind}: registry and inventory differ")
    if not set(models.__all__).issubset({f["symbol"] for f in inventory}):
        errors.append("public estimator exports missing from inventory")

    for folder in ["statistical_reference", "neural_reference"]:
        for path in (docs / folder).rglob("*.rst"):
            # Index/architecture inventories organize features, rather than
            # defining a callable. Every remaining feature requires an example.
            text = path.read_text()
            if (
                path.parent.name == "controls"
                and ":orphan:" in text
                and ":ref:" in text
            ) or path.name in {
                "inference.rst",
                "asymmetric_fitting.rst",
                "quantile_calibration.rst",
                "penalties.rst",
                "theory.rst",
            }:
                continue
            if path.parent.name == "controls" and path.stem in {
                "coefficient_fitting",
                "smoothing_optimization",
                "shape_fitting",
                "execution",
                "basis_penalty",
            }:
                continue
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
    for method in OPTIONAL_HOOKS:
        text = (docs / f"neural_reference/methods/{method}.rst").read_text()
        for name in supported_architectures(method):
            if f":doc:`{name} <../{name}>`" not in text:
                errors.append(f"{method}: missing supported architecture {name}")
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
        if (
            record.get("status") != "passed"
            or record.get("sha256") != digest
            or record.get("implementation_sha256") != implementation
        ):
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
