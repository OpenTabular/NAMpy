"""Reusable observation-space transforms for correlated additive models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ObservationTransform(Protocol):
    """Left transform applied consistently to response and design rows."""

    @property
    def size(self) -> int: ...

    @property
    def is_identity(self) -> bool: ...

    def apply(self, values) -> np.ndarray: ...

    def transform_system(self, design, response, offset=None): ...


@dataclass(frozen=True)
class IdentityObservationTransform:
    size: int

    def __post_init__(self) -> None:
        if int(self.size) < 0:
            raise ValueError("Observation-transform size must be non-negative.")
        object.__setattr__(self, "size", int(self.size))

    @property
    def is_identity(self) -> bool:
        return True

    def apply(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != self.size:
            raise ValueError(
                f"Observation array has {array.shape[0]} rows, expected {self.size}."
            )
        return array.copy()

    def transform_system(self, design, response, offset=None):
        X = self.apply(design)
        y = self.apply(response)
        off = np.zeros(self.size, dtype=np.float64) if offset is None else self.apply(offset)
        return X, y, off


@dataclass(frozen=True)
class AR1ObservationTransform:
    """SCAM/mgcv-style inverse-root transform for AR(1) residuals."""

    size: int
    rho: float
    starts: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = int(self.size)
        rho = float(self.rho)
        if size < 0:
            raise ValueError("Observation-transform size must be non-negative.")
        if not -1.0 < rho < 1.0:
            raise ValueError("AR1 rho must be strictly between -1 and 1.")
        starts = (
            np.zeros(size, dtype=bool)
            if self.starts is None
            else np.asarray(self.starts, dtype=bool).reshape(-1).copy()
        )
        if starts.shape != (size,):
            raise ValueError(f"AR1 starts must have shape ({size},), got {starts.shape}.")
        starts.setflags(write=False)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "starts", starts)

    @property
    def is_identity(self) -> bool:
        return self.rho == 0.0

    def apply(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[0] != self.size:
            raise ValueError(
                f"Observation array must have first dimension {self.size}, got {array.shape}."
            )
        out = array.copy()
        if self.size == 0 or self.rho == 0.0:
            return out
        ld = 1.0 / np.sqrt(1.0 - self.rho**2)
        sd = -self.rho * ld
        for index in range(1, self.size):
            if self.starts[index]:
                out[index] = array[index]
            else:
                out[index] = sd * array[index - 1] + ld * array[index]
        return out

    def transform_system(self, design, response, offset=None):
        X = self.apply(design)
        y = np.asarray(response, dtype=np.float64)
        off = (
            np.zeros(self.size, dtype=np.float64)
            if offset is None
            else np.asarray(offset, dtype=np.float64).reshape(-1)
        )
        # Transform the complete linear relation y = offset + X beta + error.
        return X, self.apply(y - off), np.zeros(self.size, dtype=np.float64)


def make_observation_transform(*, size: int, ar1_rho=0.0, ar_start=None):
    rho = float(ar1_rho)
    if rho == 0.0:
        return IdentityObservationTransform(size)
    return AR1ObservationTransform(size=size, rho=rho, starts=ar_start)


__all__ = [
    "AR1ObservationTransform",
    "IdentityObservationTransform",
    "ObservationTransform",
    "make_observation_transform",
]
