"""Capability discovery shared by NAMpy estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Capabilities:
    """Feature support advertised by a model or estimator."""

    supports_predict_proba: bool
    supports_standard_errors: bool
    supports_lpmatrix: bool
    supports_term_contributions: bool


@runtime_checkable
class SupportsCapabilities(Protocol):
    """Structural contract for objects that advertise backend capabilities."""

    def capabilities(self) -> Capabilities:
        """Return the capabilities supported by this object."""
        ...
