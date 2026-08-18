"""Persistence contracts shared by NAMpy estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

PersistableT = TypeVar("PersistableT", bound="PersistableModel")


@runtime_checkable
class PersistableModel(Protocol):
    """Structural contract for models supporting pickle-style persistence."""

    def save_model(self, path: str | Path) -> Path:
        """Save the model and return the destination path."""
        ...

    @classmethod
    def load_model(
        cls: type[PersistableT], path: str | Path
    ) -> PersistableT:
        """Load and return a model of the receiving class."""
        ...
