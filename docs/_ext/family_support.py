"""Generate user-facing family support from the owning backend registries."""

from pathlib import Path


def support_rows():
    from nampy.gam.families.registry import GENERAL_FAMILIES
    from nampy.neural.families.registry import FAMILY_REGISTRY

    statistical = {
        spec.public_name or spec.name: spec for spec in GENERAL_FAMILIES.values()
    }
    for name in sorted(set(statistical) | set(FAMILY_REGISTRY)):
        gam, neural = statistical.get(name), FAMILY_REGISTRY.get(name)

        def probability(spec):
            if spec is None or not spec.supports_distribution_prediction:
                return "Unavailable"
            law = spec.law_factory() if spec.law_factory is not None else None
            return (
                ", ".join(sorted(law.spec.operations))
                if law is not None
                else "Adapter declared"
            )

        yield (
            name,
            gam.name if gam else "Unavailable",
            neural.name if neural else "Unavailable",
            probability(gam),
            probability(neural),
        )


def generate(root):
    rows = list(support_rows())
    lines = [
        ".. Generated from statistical and neural family declarations; do not edit.",
        "",
        ".. list-table:: Distributional fitting families and probability queries",
        "   :header-rows: 1",
        "",
        "   * - Family",
        "     - Statistical GAMLSS name",
        "     - Neural objective name",
        "     - Statistical probability queries",
        "     - Neural probability queries",
    ]
    for row in rows:
        lines.append("   * - ``" + row[0] + "``")
        lines.extend("     - " + cell for cell in row[1:])
    destination = Path(root) / "docs/_generated/family_support.inc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return rows


def setup(app):
    app.connect("builder-inited", lambda app: generate(Path(app.srcdir).parent))
    return {"parallel_read_safe": True, "parallel_write_safe": True}
