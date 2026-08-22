from __future__ import annotations

import pytest

from tests import mgcv_parity_utils

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_missing_cache_entry_allows_live_regeneration_when_r_is_available(
    monkeypatch, tmp_path
):
    """
    Cache-behavior coverage verifying that missing cache entry allows live regeneration
    when r is available.
    """
    monkeypatch.delenv("MGCV_CACHE_ONLY", raising=False)
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "R_SCRIPT", "Rscript")

    assert mgcv_parity_utils._mgcv_cache_load("missing") is None


def test_missing_cache_entry_respects_explicit_cache_only_override(
    monkeypatch, tmp_path
):
    """
    Cache-behavior coverage verifying that missing cache entry respects explicit cache
    only override.
    """
    monkeypatch.setenv("MGCV_CACHE_ONLY", "1")
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "R_SCRIPT", "Rscript")

    with pytest.raises(RuntimeError, match="cache-only mode is enabled"):
        mgcv_parity_utils._mgcv_cache_load("missing")


def test_missing_cache_entry_raises_when_r_is_unavailable(monkeypatch, tmp_path):
    """
    Cache-behavior coverage verifying that missing cache entry raises when r is
    unavailable.
    """
    monkeypatch.delenv("MGCV_CACHE_ONLY", raising=False)
    monkeypatch.setattr(mgcv_parity_utils, "_MGCV_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mgcv_parity_utils, "R_SCRIPT", None)

    with pytest.raises(RuntimeError, match="R is not available"):
        mgcv_parity_utils._mgcv_cache_load("missing")
