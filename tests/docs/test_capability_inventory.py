"""Support declarations are live; validation annotations are scoped evidence."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = runpy.run_path(str(ROOT / "docs/_ext/capability_inventory.py"))


@pytest.fixture(scope="module")
def inventory():
    return INVENTORY["inventory"]()


def test_inventory_families_follow_runtime_registries(inventory):
    from nampy.gam.families.registry import GENERAL_FAMILIES
    from nampy.neural.families.registry import FAMILY_REGISTRY

    indexed = {(r["backend"], r["family"]): r for r in inventory["families"]}
    assert set(indexed) == (
        {("statistical", name) for name in GENERAL_FAMILIES}
        | {("neural", name) for name in FAMILY_REGISTRY}
    )
    for backend, registry in (("statistical", GENERAL_FAMILIES), ("neural", FAMILY_REGISTRY)):
        for name, spec in registry.items():
            law = spec.law_factory() if spec.law_factory is not None else None
            assert indexed[backend, name]["probability_operations"] == (
                sorted(law.spec.operations) if law is not None else []
            )
    assert indexed["neural", "quantile"]["probability_operations"] == []
    assert indexed["statistical", "beinf_lss"]["observation_type"] == "mixed"
    assert "censored" in indexed["statistical", "normal_observation_lss"]["observation_type"]


def test_inventory_uncertainty_is_live_and_calibrated_quantiles_are_distinct(inventory):
    from nampy import CalibratedQuantileGAM, QuantileGAM

    indexed = {r["model"]: r for r in inventory["models"]}
    key = "uncertainty.quantile_function"
    for cls in (QuantileGAM, CalibratedQuantileGAM):
        expected = cls().capabilities()[key]
        actual = indexed[cls.__name__]["uncertainty_methods"][key]
        assert actual["status"] == expected.state
        assert actual["conditions"] == list(expected.conditions)
    assert indexed["QuantileGAM"]["uncertainty_methods"][key]["status"] == "unsupported"
    assert indexed["CalibratedQuantileGAM"]["uncertainty_methods"][key]["status"] == "conditional"


def test_evidence_is_independent_scoped_and_links_real_tests(inventory):
    for entry in [*inventory["models"], *inventory["families"]]:
        assert "validation_evidence" in entry
        assert "validated" not in entry
        for evidence in entry["validation_evidence"]:
            assert evidence["scope"].startswith(("Selected", "qgam"))
            assert (ROOT / evidence["test_module"]).is_file()
    # An unindexed case keeps its declarations and does not acquire a spurious
    # evidence claim from another family merely because their APIs overlap.
    entry = next(r for r in inventory["families"] if r["backend"] == "neural" and r["family"] == "normal")
    assert "log_prob" in entry["probability_operations"]
    assert entry["validation_evidence"] == []


def test_generation_is_deterministic_and_checked_in(tmp_path):
    generated = INVENTORY["generate"](tmp_path)
    target = "docs/_generated/capabilities"
    assert json.loads((tmp_path / target / "inventory.json").read_text()) == generated
    for path in (tmp_path / target).iterdir():
        assert path.read_text() == (ROOT / target / path.name).read_text()


def test_expectile_route_reports_its_actual_loss_and_support(inventory):
    from nampy import ExpectileGAM

    row = next(r for r in inventory["models"] if r["model"] == "ExpectileGAM")
    assert "LAWS" in row["fitting_route"]
    assert "no automatic selection" in row["smoothing_selection"]
    actual = ExpectileGAM().capabilities()
    for key in ("predict_distribution", "predict_quantiles"):
        assert row["probability_operations"][key]["status"] == actual[key].state == "unsupported"
    assert row["uncertainty_methods"]["uncertainty.coefficients"]["status"] == "unsupported"
    assert any("weighted expectile" in e["scope"] for e in row["validation_evidence"])
