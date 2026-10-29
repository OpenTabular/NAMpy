"""Representation choices for identifiable structured/neural composition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultSemiStructuredConfig:
    structured_formula: str | None = None
    neural_features: tuple[str, ...] = ()
    hidden_dims: tuple[int, ...] = (16,)
    representation_dim: int = 8
    k: int = 6
    basis: str = "tp"

    def __post_init__(self):
        if self.structured_formula is not None and (
            not isinstance(self.structured_formula, str)
            or not self.structured_formula.lstrip().startswith("~")
        ):
            raise ValueError("structured_formula must be a right-hand-side formula.")
        if isinstance(self.neural_features, str) or len(
            set(self.neural_features)
        ) != len(self.neural_features):
            raise ValueError("neural_features must be distinct source feature names.")
        if self.representation_dim < 1 or any(value < 1 for value in self.hidden_dims):
            raise ValueError("Neural representation dimensions must be positive.")
        object.__setattr__(self, "neural_features", tuple(self.neural_features))
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
