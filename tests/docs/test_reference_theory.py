"""Require authored theory when a public statistical feature is registered."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))


def test_theory_covers_the_live_feature_registries():
    from family_math import distribution_contract
    from family_theory import family_theory
    from feature_reference import FAMILY_SUPPORT, OPTIMIZER_MEANINGS, SELECTION_MEANINGS
    from reference_theory import smooth_theory
    from selection_theory import selection_theory

    import nampy.gam.smooths  # noqa: F401
    import nampy.gam.specs  # noqa: F401
    from nampy.gam.basis_registry import basis_descriptors
    from nampy.gam.families.registry import GENERAL_FAMILIES
    from nampy.neural.distributions.registry import FAMILY_REGISTRY

    for descriptor in basis_descriptors():
        assert ".. math::" in smooth_theory(descriptor.name)
    for name in set(FAMILY_SUPPORT) | set(GENERAL_FAMILIES):
        assert ".. math::" in family_theory(name, "statistical")
    for name in FAMILY_REGISTRY:
        assert ".. math::" in family_theory(name, "neural")
    for backend, names in [
        ("statistical", set(FAMILY_SUPPORT) | set(GENERAL_FAMILIES)),
        ("neural", set(FAMILY_REGISTRY)),
    ]:
        for name in names:
            contract = distribution_contract(name, backend)
            assert len(contract) == 4
            for index, value in enumerate(contract):
                if name == "quantile" and index >= 2:
                    continue  # Quantile outputs do not determine these moments.
                assert ":math:`" in value, (backend, name, index)
    for name in SELECTION_MEANINGS | OPTIMIZER_MEANINGS:
        assert ".. math::" in selection_theory(name)


@pytest.mark.parametrize("kind", ["smooth", "family", "family_math", "selection"])
def test_unknown_features_require_an_authored_theory_decision(kind):
    from family_math import distribution_contract
    from family_theory import family_theory
    from reference_theory import smooth_theory
    from selection_theory import selection_theory

    render = {
        "smooth": smooth_theory,
        "family": lambda name: family_theory(name, "statistical"),
        "family_math": lambda name: distribution_contract(name, "statistical"),
        "selection": selection_theory,
    }[kind]
    with pytest.raises(KeyError):
        render("not_an_authored_feature")
