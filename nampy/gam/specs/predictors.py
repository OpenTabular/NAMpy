from dataclasses import dataclass, field
from typing import Any


@dataclass
class LinearPredictorSpec:
    name: str = "eta"
    terms: list = field(default_factory=list)
    has_intercept: bool = False
    parameter_name: str | None = None
    offset_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
