from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests import mgcv_parity_utils
from tests.reference_fixtures import (
    AliasedReferenceKey,
    FixtureIdentity,
    load_reference,
    reference_key,
    save_reference,
)

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_missing_fixture_fails_even_when_r_is_available(monkeypatch, tmp_path):
    """Normal parity tests never invoke an available upstream implicitly."""
    monkeypatch.delenv("NAMPY_REFRESH_REFERENCE_FIXTURES", raising=False)
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "R_SCRIPT", "Rscript")

    with pytest.raises(RuntimeError, match="committed static mgcv"):
        mgcv_parity_utils._mgcv_fixture_load("missing")


def test_missing_fixture_allows_explicit_local_refresh(monkeypatch, tmp_path):
    """Only the explicit fixture-refresh mode may fall through to upstream."""
    monkeypatch.setenv("NAMPY_REFRESH_REFERENCE_FIXTURES", "1")
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "R_SCRIPT", "Rscript")

    assert mgcv_parity_utils._mgcv_fixture_load("missing") is None


def test_explicit_refresh_writes_a_deterministic_fixture(monkeypatch, tmp_path):
    """Generated fixtures round-trip through compressed JSON."""
    monkeypatch.setenv("NAMPY_REFRESH_REFERENCE_FIXTURES", "1")
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_DIR", tmp_path)

    expected = {"value": [1.0, 2.0], "label": "mgcv"}
    mgcv_parity_utils._mgcv_fixture_save("example", expected)

    assert mgcv_parity_utils._mgcv_fixture_load("example") == expected
    assert (tmp_path / "example.json.gz").is_file()


def test_explicit_rebuild_bypasses_an_existing_fixture(monkeypatch, tmp_path):
    """A source-version rebaseline can deliberately replace existing data."""
    monkeypatch.setenv("NAMPY_REFRESH_REFERENCE_FIXTURES", "1")
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_DIR", tmp_path)
    mgcv_parity_utils._mgcv_fixture_save("example", {"version": 1})

    monkeypatch.delenv("NAMPY_REFRESH_REFERENCE_FIXTURES")
    monkeypatch.setenv("NAMPY_REBUILD_REFERENCE_FIXTURES", "1")
    assert mgcv_parity_utils._mgcv_fixture_load("example") is None

    mgcv_parity_utils._mgcv_fixture_save("example", {"version": 2})
    monkeypatch.delenv("NAMPY_REBUILD_REFERENCE_FIXTURES")
    assert mgcv_parity_utils._mgcv_fixture_load("example") == {"version": 2}


def test_portable_dataframe_fixture_identity_ignores_final_bit_platform_noise():
    """Equivalent libm results share a static raw-constructor fixture key."""
    value = 0.912763940260521
    platform_neighbor = np.nextafter(value, np.inf)

    left = pd.DataFrame({"x": [value], "label": ["a"]})
    right = pd.DataFrame({"x": [platform_neighbor], "label": ["a"]})
    meaningfully_different = pd.DataFrame({"x": [value + 1e-10], "label": ["a"]})

    assert mgcv_parity_utils._portable_df_fixture_repr(
        left
    ) == mgcv_parity_utils._portable_df_fixture_repr(right)
    assert mgcv_parity_utils._portable_df_fixture_repr(
        left
    ) != mgcv_parity_utils._portable_df_fixture_repr(meaningfully_different)


def test_portable_fixture_key_records_and_reuses_existing_fixture_alias(
    monkeypatch, tmp_path
):
    """Canonical keys reuse committed fixtures created with historical text."""
    aliases_path = tmp_path / "aliases.json"
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_FIXTURE_ALIASES", aliases_path)
    key = mgcv_parity_utils._mgcv_fixture_key(
        "example",
        {"data": FixtureIdentity("x\n0.5\n", "x\n0.5000000000000001\n")},
    )
    assert key.legacy is not None

    monkeypatch.setenv("NAMPY_REFRESH_REFERENCE_FIXTURES", "1")
    mgcv_parity_utils._mgcv_fixture_save(key.legacy, {"value": 3})
    monkeypatch.delenv("NAMPY_REFRESH_REFERENCE_FIXTURES")
    monkeypatch.setenv("NAMPY_RECORD_REFERENCE_ALIASES", "1")
    assert mgcv_parity_utils._mgcv_fixture_load(key) == {"value": 3}
    assert json.loads(aliases_path.read_text(encoding="utf-8")) == {
        str(key): key.legacy
    }

    monkeypatch.delenv("NAMPY_RECORD_REFERENCE_ALIASES")
    alias_only_key = AliasedReferenceKey(str(key), "unavailable")
    assert mgcv_parity_utils._mgcv_fixture_load(alias_only_key) == {"value": 3}


def test_direct_reference_key_records_and_reuses_legacy_float_payload(
    monkeypatch, tmp_path
):
    """Direct family helpers migrate their historical numeric fixture keys."""
    value = 0.912763940260521
    key = reference_key(
        "example",
        {"values": [np.nextafter(value, np.inf)]},
        normalize_floats=True,
    )
    assert isinstance(key, AliasedReferenceKey)
    assert key.legacy is not None

    monkeypatch.setenv("NAMPY_REFRESH_REFERENCE_FIXTURES", "1")
    save_reference("mgcv", key.legacy, {"value": 7}, root=tmp_path)
    monkeypatch.delenv("NAMPY_REFRESH_REFERENCE_FIXTURES")
    monkeypatch.setenv("NAMPY_RECORD_REFERENCE_ALIASES", "1")

    assert load_reference("mgcv", key, root=tmp_path) == {"value": 7}
    aliases_path = tmp_path / "mgcv" / "aliases.json"
    assert json.loads(aliases_path.read_text(encoding="utf-8")) == {
        str(key): key.legacy
    }

    monkeypatch.delenv("NAMPY_RECORD_REFERENCE_ALIASES")
    alias_only_key = AliasedReferenceKey(str(key), "unavailable")
    assert load_reference("mgcv", alias_only_key, root=tmp_path) == {"value": 7}


def test_committed_mgcv_fixture_aliases_resolve_to_static_fixtures():
    """Every portable alias remains usable on machines without R."""
    aliases = json.loads(
        mgcv_parity_utils._MGCV_FIXTURE_ALIASES.read_text(encoding="utf-8")
    )
    missing = [
        target
        for target in aliases.values()
        if not (mgcv_parity_utils._MGCV_FIXTURE_DIR / f"{target}.json.gz").is_file()
    ]

    assert not missing


def test_committed_scam_fixture_aliases_resolve_to_static_fixtures():
    """Every portable SCAM alias remains usable without a live R package."""
    fixture_dir = mgcv_parity_utils._MGCV_FIXTURE_DIR.parent / "scam"
    aliases = json.loads((fixture_dir / "aliases.json").read_text(encoding="utf-8"))
    missing = [
        target
        for target in aliases.values()
        if not (fixture_dir / f"{target}.json.gz").is_file()
    ]

    assert not missing


def test_raw_constructor_fixture_identity_uses_only_referenced_columns():
    """Response and unrelated columns cannot perturb a constructor fixture key."""
    data = pd.DataFrame(
        {
            "y": [0.1, 0.2],
            "x": [1.0, 2.0],
            "by_factor": ["a", "b"],
            "unrelated": [3.0, 4.0],
        }
    )

    selected = mgcv_parity_utils._raw_constructor_fixture_frame(
        data,
        's(x, by=by_factor, bs="cr")',
    )

    assert list(selected.columns) == ["x", "by_factor"]
