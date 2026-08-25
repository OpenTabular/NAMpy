"""Declarative registry for GAM smooth-basis construction capabilities.

The registry is deliberately independent of both the formula-specification and
runtime-term packages.  Those layers attach their builder/class at import time,
while composition code consumes the immutable capability metadata defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class SmoothBasisDescriptor:
    """Construction and composition contract for one ``s(..., bs=)`` basis."""

    name: str
    direct_runtime: bool = False
    min_features: int = 1
    max_features: int | None = 1
    supports_pc: bool = False
    supports_tensor: bool = False
    supports_factor_smooth: bool = False
    supports_fs: bool = True
    accepts_xt: bool = False
    factor_accepts_xt: bool = False
    factor_forwards_m: bool = False
    dynamic_default_k: bool = False
    runtime_options: tuple[str, ...] = ()
    runtime_class: type | None = None
    spec_builder: Callable[[dict[str, Any]], Any] | None = None

    def validate_feature_count(self, count: int, *, context: str) -> None:
        count = int(count)
        if count < self.min_features or (
            self.max_features is not None and count > self.max_features
        ):
            if self.min_features == self.max_features:
                expected = f"exactly {self.min_features}"
            elif self.max_features is None:
                expected = f"at least {self.min_features}"
            else:
                expected = f"between {self.min_features} and {self.max_features}"
            raise ValueError(
                f"{context} basis {self.name!r} requires {expected} feature(s), "
                f"got {count}."
            )


def _descriptor(
    name: str, *, direct_runtime: bool = True, **kwargs: Any
) -> SmoothBasisDescriptor:
    return SmoothBasisDescriptor(
        name=str(name).lower(), direct_runtime=direct_runtime, **kwargs
    )


# This catalog is the single source of truth for regular metric/structured
# bases. Formula-only special bases (shape constraints, re/fs/sz) are attached
# lazily with ``register_basis_spec_builder`` below.
_BASIS_REGISTRY: dict[str, SmoothBasisDescriptor] = {
    "ad": _descriptor(
        "ad",
        min_features=1,
        max_features=2,
        supports_pc=True,
        supports_factor_smooth=False,
        supports_fs=False,
        accepts_xt=True,
        dynamic_default_k=True,
        runtime_options=("m", "xt", "pc"),
    ),
    "cr": _descriptor(
        "cr",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        runtime_options=("basis", "pc", "shared_basis_setup"),
    ),
    "cs": _descriptor(
        "cs",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        supports_fs=False,
        runtime_options=("basis", "pc", "shared_basis_setup"),
    ),
    "cc": _descriptor(
        "cc",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        runtime_options=("basis", "pc", "shared_basis_setup"),
    ),
    "ps": _descriptor(
        "ps",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        runtime_options=("basis", "m", "pc"),
    ),
    "cp": _descriptor(
        "cp",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        runtime_options=("basis", "m", "pc"),
    ),
    "bs": _descriptor(
        "bs",
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        dynamic_default_k=True,
        runtime_options=("m", "pc"),
    ),
    "ds": _descriptor(
        "ds",
        max_features=None,
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        accepts_xt=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        dynamic_default_k=True,
        runtime_options=("m", "xt", "pc"),
    ),
    "gp": _descriptor(
        "gp",
        max_features=None,
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        accepts_xt=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        dynamic_default_k=True,
        runtime_options=("m", "xt", "pc"),
    ),
    "sos": _descriptor(
        "sos",
        min_features=2,
        max_features=2,
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        accepts_xt=True,
        factor_accepts_xt=True,
        factor_forwards_m=True,
        dynamic_default_k=True,
        runtime_options=("m", "xt", "pc"),
    ),
    "mrf": _descriptor(
        "mrf",
        supports_tensor=True,
        supports_factor_smooth=True,
        accepts_xt=True,
        factor_accepts_xt=True,
        dynamic_default_k=True,
        runtime_options=("xt",),
    ),
    "tp": _descriptor(
        "tp",
        max_features=None,
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        accepts_xt=True,
        factor_accepts_xt=True,
        dynamic_default_k=True,
        runtime_options=("basis", "m", "xt", "pc"),
    ),
    "ts": _descriptor(
        "ts",
        max_features=None,
        supports_pc=True,
        supports_tensor=True,
        supports_factor_smooth=True,
        supports_fs=False,
        accepts_xt=True,
        factor_accepts_xt=True,
        dynamic_default_k=True,
        runtime_options=("basis", "m", "xt", "pc"),
    ),
}


def get_basis_descriptor(name: str) -> SmoothBasisDescriptor | None:
    return _BASIS_REGISTRY.get(str(name).lower())


def require_basis_descriptor(name: str) -> SmoothBasisDescriptor:
    descriptor = get_basis_descriptor(name)
    if descriptor is None:
        raise ValueError(f"Unknown smooth basis {name!r}.")
    return descriptor


def basis_descriptors() -> tuple[SmoothBasisDescriptor, ...]:
    return tuple(_BASIS_REGISTRY.values())


def tensor_basis_names() -> frozenset[str]:
    return frozenset(
        descriptor.name
        for descriptor in _BASIS_REGISTRY.values()
        if descriptor.supports_tensor
    )


def register_basis_runtime(name: str, runtime_class: type) -> type:
    key = str(name).lower()
    descriptor = _BASIS_REGISTRY.get(key, _descriptor(key, direct_runtime=False))
    existing = descriptor.runtime_class
    if existing is not None and existing is not runtime_class:
        raise ValueError(f"Smooth basis {key!r} already has a runtime class.")
    _BASIS_REGISTRY[key] = replace(descriptor, runtime_class=runtime_class)
    return runtime_class


def register_basis_spec_builder(
    name: str, builder: Callable[[dict[str, Any]], Any]
) -> Callable[[dict[str, Any]], Any]:
    key = str(name).lower()
    descriptor = _BASIS_REGISTRY.get(key, _descriptor(key, direct_runtime=False))
    existing = descriptor.spec_builder
    if existing is not None and existing is not builder:
        raise ValueError(f"Smooth basis {key!r} already has a spec builder.")
    _BASIS_REGISTRY[key] = replace(descriptor, spec_builder=builder)
    return builder


def basis_spec_builder(*names: str):
    """Attach a specification builder to one or more basis descriptors."""
    if not names:
        raise ValueError("At least one smooth basis name is required.")

    def decorator(builder: Callable[[dict[str, Any]], Any]):
        for name in names:
            register_basis_spec_builder(name, builder)
        return builder

    return decorator


__all__ = [
    "SmoothBasisDescriptor",
    "basis_descriptors",
    "basis_spec_builder",
    "get_basis_descriptor",
    "register_basis_runtime",
    "register_basis_spec_builder",
    "require_basis_descriptor",
    "tensor_basis_names",
]
