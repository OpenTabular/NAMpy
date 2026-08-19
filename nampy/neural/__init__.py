"""Torch-side backend internals, imported on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "configs",
    "data",
    "distributions",
    "modules",
    "contracts",
    "task",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f".{name}", __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
