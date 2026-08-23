"""Static upstream-reference fixtures used by parity tests.

Normal tests are deliberately unable to execute an upstream implementation.
Developers may opt into fixture generation explicitly with
``NAMPY_REFRESH_REFERENCE_FIXTURES=1`` while the relevant upstream source or
package is available in their local environment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from tests._paths import TESTS_DIR

FIXTURE_ROOT = TESTS_DIR / "reference_fixtures"
REFRESH_ENV = "NAMPY_REFRESH_REFERENCE_FIXTURES"
REBUILD_ENV = "NAMPY_REBUILD_REFERENCE_FIXTURES"
RECORD_ALIASES_ENV = "NAMPY_RECORD_REFERENCE_ALIASES"


class MissingReferenceFixture(RuntimeError):
    """Raised when a parity case has no committed static reference result."""


class FixtureIdentity(str):
    """Canonical fixture input text paired with its pre-migration spelling."""

    legacy: str

    def __new__(cls, canonical: str, legacy: str):
        value = super().__new__(cls, canonical)
        value.legacy = legacy
        return value


class AliasedReferenceKey(str):
    """Canonical fixture key paired with the key used by existing fixtures."""

    legacy: str | None

    def __new__(cls, canonical: str, legacy: str | None = None):
        value = super().__new__(cls, canonical)
        value.legacy = legacy
        return value


def refresh_enabled() -> bool:
    """Return whether generation of missing local fixtures is enabled."""
    return os.environ.get(REFRESH_ENV, "").lower() in {"1", "true", "yes", "on"}


def rebuild_enabled() -> bool:
    """Return whether existing fixtures should be regenerated from upstream."""
    return os.environ.get(REBUILD_ENV, "").lower() in {"1", "true", "yes", "on"}


def generation_enabled() -> bool:
    """Return whether fixture writes are explicitly authorized."""
    return refresh_enabled() or rebuild_enabled()


def reference_key(operation: str, payload: Any) -> str:
    """Build a stable content key for one upstream operation and its inputs."""
    encoded = json.dumps(
        {"operation": operation, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _map_fixture_payload(value: Any, *, legacy: bool, normalize_floats: bool) -> Any:
    """Map a fixture payload to its canonical or historical representation."""
    if isinstance(value, FixtureIdentity):
        return value.legacy if legacy else str(value)
    if isinstance(value, dict):
        return {
            key: _map_fixture_payload(
                item, legacy=legacy, normalize_floats=normalize_floats
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _map_fixture_payload(
                item, legacy=legacy, normalize_floats=normalize_floats
            )
            for item in value
        ]
    if normalize_floats and not legacy and isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def aliased_reference_key(
    operation: str,
    payload: Any,
    *,
    normalize_floats: bool = False,
) -> AliasedReferenceKey:
    """Build a portable key while retaining the existing fixture key."""
    canonical_payload = _map_fixture_payload(
        payload, legacy=False, normalize_floats=normalize_floats
    )
    legacy_payload = _map_fixture_payload(
        payload, legacy=True, normalize_floats=normalize_floats
    )
    canonical = reference_key(operation, canonical_payload)
    legacy = reference_key(operation, legacy_payload)
    return AliasedReferenceKey(canonical, None if legacy == canonical else legacy)


def fixture_payload_variants(
    payload: Any,
    *,
    normalize_floats: bool = False,
) -> tuple[Any, Any]:
    """Return canonical and historical forms of a fixture-key payload."""
    return (
        _map_fixture_payload(
            payload, legacy=False, normalize_floats=normalize_floats
        ),
        _map_fixture_payload(
            payload, legacy=True, normalize_floats=normalize_floats
        ),
    )


def portable_dataframe_repr(frame: pd.DataFrame) -> str:
    """Return a platform-stable DataFrame identity for fixture keys.

    NumPy transcendental functions can differ by a final binary digit across
    system math libraries. Pandas' default full-precision CSV output preserves
    that immaterial noise and consequently gives equivalent parity inputs
    different fixture keys. Twelve significant decimal digits remain finer
    than the data-scale parity tolerances while avoiding decimal rounding
    boundaries that a one-ULP difference can cross at fifteen digits.
    """
    return frame.to_csv(
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )


def portable_dataframe_identity(
    frame: pd.DataFrame,
    *,
    legacy_float_format: str | None = None,
) -> FixtureIdentity:
    """Return portable DataFrame text with its historical representation."""
    legacy = frame.to_csv(
        index=False,
        float_format=legacy_float_format,
        lineterminator="\n",
    )
    return FixtureIdentity(portable_dataframe_repr(frame), legacy)


def load_aliased_reference(
    namespace: str,
    key: str,
    *,
    aliases_path: Path,
    root: Path | None = None,
) -> Any | None:
    """Load a canonical fixture through a committed legacy-key alias."""
    try:
        return load_reference(namespace, str(key), root=root)
    except MissingReferenceFixture as canonical_error:
        aliases = (
            json.loads(aliases_path.read_text(encoding="utf-8"))
            if aliases_path.exists()
            else {}
        )
        target = aliases.get(str(key))
        legacy = getattr(key, "legacy", None)
        if target is None and legacy is not None:
            target = legacy
        if target is None:
            raise canonical_error
        try:
            result = load_reference(namespace, target, root=root)
        except MissingReferenceFixture:
            raise canonical_error from None
        if (
            target == legacy
            and aliases.get(str(key)) != target
            and os.environ.get(RECORD_ALIASES_ENV, "").lower()
            in {"1", "true", "yes", "on"}
        ):
            aliases[str(key)] = target
            aliases_path.parent.mkdir(parents=True, exist_ok=True)
            aliases_path.write_text(
                json.dumps(aliases, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return result


def fixture_path(namespace: str, key: str, *, root: Path | None = None) -> Path:
    """Return the compressed JSON path for a namespaced fixture key."""
    base = FIXTURE_ROOT if root is None else root
    return base / namespace / f"{key}.json.gz"


def load_reference(
    namespace: str,
    key: str,
    *,
    root: Path | None = None,
) -> Any | None:
    """Load a fixture, or return ``None`` only in explicit refresh mode."""
    path = fixture_path(namespace, key, root=root)
    if rebuild_enabled():
        return None
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if generation_enabled():
        return None
    raise MissingReferenceFixture(
        f"Missing static {namespace} reference fixture {key} at {path}. "
        f"Generate it locally with {REFRESH_ENV}=1 and commit the result."
    )


def save_reference(
    namespace: str,
    key: str,
    value: Any,
    *,
    root: Path | None = None,
) -> Path:
    """Persist deterministic compressed JSON during explicit refresh mode."""
    if not generation_enabled():
        raise RuntimeError(
            "Refusing to write reference fixtures unless generation is explicitly "
            f"enabled with {REFRESH_ENV}=1 or {REBUILD_ENV}=1."
        )
    path = fixture_path(namespace, key, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return path


__all__ = [
    "AliasedReferenceKey",
    "FIXTURE_ROOT",
    "FixtureIdentity",
    "MissingReferenceFixture",
    "REBUILD_ENV",
    "REFRESH_ENV",
    "RECORD_ALIASES_ENV",
    "aliased_reference_key",
    "fixture_path",
    "fixture_payload_variants",
    "generation_enabled",
    "load_reference",
    "load_aliased_reference",
    "portable_dataframe_identity",
    "portable_dataframe_repr",
    "reference_key",
    "rebuild_enabled",
    "refresh_enabled",
    "save_reference",
]
