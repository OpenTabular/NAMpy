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

from tests._paths import TESTS_DIR

FIXTURE_ROOT = TESTS_DIR / "reference_fixtures"
REFRESH_ENV = "NAMPY_REFRESH_REFERENCE_FIXTURES"
REBUILD_ENV = "NAMPY_REBUILD_REFERENCE_FIXTURES"


class MissingReferenceFixture(RuntimeError):
    """Raised when a parity case has no committed static reference result."""


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
    "FIXTURE_ROOT",
    "MissingReferenceFixture",
    "REBUILD_ENV",
    "REFRESH_ENV",
    "fixture_path",
    "generation_enabled",
    "load_reference",
    "reference_key",
    "rebuild_enabled",
    "refresh_enabled",
    "save_reference",
]
