"""Versioned pickle-state handling for fitted GAM models."""

from __future__ import annotations

from typing import Any

GAM_STATE_SCHEMA_VERSION = 3
_SCHEMA_KEY = "_gam_state_schema_version"


def gam_pickle_state(model: Any) -> dict[str, Any]:
    state = dict(model.__dict__)
    # Warm starts and evaluation caches are private to one fit invocation.
    state.pop("_ws", None)
    state[_SCHEMA_KEY] = GAM_STATE_SCHEMA_VERSION
    return state


def restore_gam_pickle_state(model: Any, state: dict[str, Any]) -> None:
    state = dict(state)
    version = int(state.pop(_SCHEMA_KEY, 0))
    if version != GAM_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported GAM state schema {version}; expected {GAM_STATE_SCHEMA_VERSION}.")
    model.__dict__.update(state)


__all__ = [
    "GAM_STATE_SCHEMA_VERSION",
    "gam_pickle_state",
    "restore_gam_pickle_state",
]
