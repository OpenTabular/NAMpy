from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


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
    basis_options: dict[str, Any] = field(default_factory=dict)
    smoothing_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    term_id: str = field(default_factory=_make_term_id)
