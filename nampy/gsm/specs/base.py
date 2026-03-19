from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class BaseTermSpec:
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, **updates):
        return replace(self, **updates)