from dataclasses import dataclass
from .base import BaseTermSpec


@dataclass
class ParametricTermSpec(BaseTermSpec):
    name: str = ""
    raw_label: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("ParametricTermSpec requires a non-empty `name`.")
        if self.raw_label is None:
            self.raw_label = self.name
        if self.label is None:
            self.label = self.raw_label