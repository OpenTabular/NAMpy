from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from .smooth import SmoothSpec


def _make_term_id() -> str:
    return uuid4().hex[:12]


@dataclass(frozen=True)
class TermSpec:
    """
    Canonical declarative term contract used across compilation.
    """

    kind: str
    features: tuple[str, ...] = ()
    by_variable: str | None = None
    smooth_spec: SmoothSpec | None = None
    basis_options: Mapping[str, Any] = field(default_factory=dict)
    smoothing_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    term_id: str = field(default_factory=_make_term_id)

    def __post_init__(self) -> None:
        from .build import smooth_spec_from_basis_options

        smooth_spec = self.smooth_spec
        basis_options = dict(self.basis_options)

        if self.kind == "smooth":
            if smooth_spec is None:
                smooth_spec = smooth_spec_from_basis_options(basis_options)
            basis_options = smooth_spec.to_basis_options()
        else:
            smooth_spec = None
            basis_options = {}

        object.__setattr__(self, "smooth_spec", smooth_spec)
        object.__setattr__(
            self,
            "basis_options",
            MappingProxyType(basis_options),
        )
