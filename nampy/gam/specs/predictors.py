from dataclasses import dataclass, field
from typing import Any

from .base import TermSpec


@dataclass
class PenaltyGroupSpec:
    smoothing_id: str
    term_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    sp_count: int | None = None
    sp_indices: list[int] = field(default_factory=list)


@dataclass
class LinearPredictorSpec:
    name: str = "eta"
    terms: list[TermSpec] = field(default_factory=list)
    has_intercept: bool = False
    parameter_name: str | None = None
    offset_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
