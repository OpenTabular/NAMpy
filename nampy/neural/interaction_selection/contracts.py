"""Contracts shared by model-agnostic interaction detectors and selectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch

Interaction = tuple[str, ...]
PredictionFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FeatureGroups:
    """Map logical source features to columns in a transformed tensor."""

    names: tuple[str, ...]
    columns: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature group names must be unique.")
        if set(self.names) != set(self.columns):
            raise ValueError("FeatureGroups names and columns must describe the same keys.")
        claimed: set[int] = set()
        for name in self.names:
            indices = tuple(self.columns[name])
            if not indices:
                raise ValueError(f"Feature group {name!r} cannot be empty.")
            if any(index < 0 for index in indices):
                raise ValueError("Feature group column indices must be non-negative.")
            overlap = claimed.intersection(indices)
            if overlap:
                raise ValueError(f"Feature groups overlap at columns {sorted(overlap)}.")
            claimed.update(indices)

    def indices(self, interaction: Sequence[str]) -> tuple[int, ...]:
        unknown = [name for name in interaction if name not in self.columns]
        if unknown:
            raise ValueError(
                f"Unknown logical features {unknown}; available: {list(self.names)}."
            )
        return tuple(
            index for name in interaction for index in self.columns[str(name)]
        )


def concatenate_feature_tensors(
    num_features: Mapping[str, torch.Tensor],
    cat_features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, FeatureGroups]:
    """Concatenate transformed tensors while retaining source-feature groups."""
    ordered = [*num_features.items(), *cat_features.items()]
    if not ordered:
        raise ValueError("Interaction selection requires at least one feature.")
    rows = int(ordered[0][1].shape[0])
    columns: dict[str, tuple[int, ...]] = {}
    tensors = []
    start = 0
    for name, tensor in ordered:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 2 or tensor.shape[0] != rows:
            raise ValueError(
                "Every transformed feature must have shape [rows, columns] with "
                "a common row count."
            )
        width = int(tensor.shape[1])
        columns[name] = tuple(range(start, start + width))
        tensors.append(tensor.float())
        start += width
    names = tuple(name for name, _ in ordered)
    return torch.cat(tensors, dim=1), FeatureGroups(names=names, columns=columns)


@dataclass(frozen=True)
class InteractionScore:
    """Aggregated detector values for one logical feature subset."""

    interaction: Interaction
    score: float
    inclusion: float = 0.0
    removal: float = 0.0
    signed_score: float = 0.0
    total_effect: float = 0.0
    n_contrasts: int = 0


@dataclass(frozen=True)
class InteractionSelectionResult:
    """Reproducible output of a hierarchical interaction search."""

    selected_terms: tuple[Interaction, ...]
    scores: tuple[InteractionScore, ...]
    candidates_by_order: Mapping[int, tuple[Interaction, ...]] = field(
        default_factory=dict
    )
    thresholds_by_order: Mapping[int, float] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()

    @property
    def selected_interactions(self) -> tuple[Interaction, ...]:
        return tuple(term for term in self.selected_terms if len(term) > 1)

    def score_for(self, interaction: Sequence[str]) -> InteractionScore:
        requested = tuple(interaction)
        for score in self.scores:
            if score.interaction == requested:
                return score
        raise KeyError(requested)

    def to_frame(self):
        """Return detector scores as a tidy pandas table."""
        import pandas as pd

        selected = set(self.selected_terms)
        columns = [
            "interaction",
            "order",
            "selected",
            "score",
            "inclusion",
            "removal",
            "signed_score",
            "total_effect",
            "n_contrasts",
        ]
        return pd.DataFrame(
            [
                {
                    "interaction": ":".join(item.interaction),
                    "order": len(item.interaction),
                    "selected": item.interaction in selected,
                    "score": item.score,
                    "inclusion": item.inclusion,
                    "removal": item.removal,
                    "signed_score": item.signed_score,
                    "total_effect": item.total_effect,
                    "n_contrasts": item.n_contrasts,
                }
                for item in self.scores
            ],
            columns=columns,
        )


@runtime_checkable
class InteractionDetector(Protocol):
    """Model-agnostic detector operating on transformed feature groups."""

    def score(
        self,
        predict: PredictionFunction,
        inputs: torch.Tensor,
        candidates: Sequence[Interaction],
        feature_groups: FeatureGroups,
    ) -> list[InteractionScore]: ...


__all__ = [
    "FeatureGroups",
    "Interaction",
    "InteractionDetector",
    "InteractionScore",
    "InteractionSelectionResult",
    "PredictionFunction",
    "concatenate_feature_tensors",
]
