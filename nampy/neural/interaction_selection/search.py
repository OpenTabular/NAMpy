"""Generic hierarchical feature-interaction selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .combinatorics import interaction_frontier
from .contracts import (
    FeatureGroups,
    InteractionDetector,
    InteractionScore,
    InteractionSelectionResult,
    PredictionFunction,
)


@dataclass(frozen=True)
class InteractionSearchConfig:
    """Controls independent of a particular interaction score."""

    max_order: int = 3
    threshold: float | Mapping[int, float] = field(
        default_factory=lambda: {2: 0.1, 3: 0.1}
    )
    threshold_mode: str = "fraction"
    heredity_fraction: float = 0.5
    max_candidates: int = 1000
    max_terms_per_order: int | None = None

    def __post_init__(self) -> None:
        if self.max_order < 1:
            raise ValueError("max_order must be at least 1.")
        if self.threshold_mode not in {"absolute", "quantile", "fraction"}:
            raise ValueError(
                "threshold_mode must be 'absolute', 'quantile', or 'fraction'."
            )
        if not 0 <= self.heredity_fraction <= 1:
            raise ValueError("heredity_fraction must lie in [0, 1].")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive.")
        if self.max_terms_per_order is not None and self.max_terms_per_order < 1:
            raise ValueError("max_terms_per_order must be positive when provided.")

    def threshold_for(self, order: int) -> float:
        if isinstance(self.threshold, Mapping):
            value = float(self.threshold.get(order, 0.0))
        else:
            value = float(self.threshold)
        if self.threshold_mode in {"quantile", "fraction"} and not 0 <= value <= 1:
            raise ValueError(
                f"{self.threshold_mode} thresholds must lie in [0, 1]."
            )
        return value


def _select_scores(
    scores: list[InteractionScore],
    *,
    threshold: float,
    mode: str,
    max_terms: int | None,
) -> list[InteractionScore]:
    ranked = sorted(scores, key=lambda item: (-item.score, item.interaction))
    if mode == "absolute":
        selected = [item for item in ranked if item.score > threshold]
    elif mode == "quantile":
        if threshold <= 0:
            selected = []
        elif threshold >= 1:
            selected = ranked
        else:
            cutoff = float(
                np.quantile([item.score for item in ranked], 1 - threshold)
            )
            selected = [item for item in ranked if item.score >= cutoff]
    else:
        count = int(math.ceil(threshold * len(ranked)))
        selected = ranked[:count]
    if max_terms is not None:
        selected = selected[:max_terms]
    return selected


def select_interactions(
    detector: InteractionDetector,
    predict: PredictionFunction,
    inputs,
    feature_groups: FeatureGroups,
    config: InteractionSearchConfig,
) -> InteractionSelectionResult:
    """Run layerwise SIAN-style selection over a detector-independent score."""
    selected = [(name,) for name in feature_groups.names]
    all_scores: list[InteractionScore] = []
    candidates_by_order = {}
    thresholds_by_order = {}

    for order in range(2, config.max_order + 1):
        candidates = interaction_frontier(
            feature_groups.names,
            selected,
            order=order,
            heredity_fraction=config.heredity_fraction,
            max_candidates=config.max_candidates,
        )
        candidates_by_order[order] = tuple(candidates)
        if not candidates:
            break
        scores = detector.score(predict, inputs, candidates, feature_groups)
        all_scores.extend(scores)
        threshold = config.threshold_for(order)
        thresholds_by_order[order] = threshold
        selected.extend(
            item.interaction
            for item in _select_scores(
                scores,
                threshold=threshold,
                mode=config.threshold_mode,
                max_terms=config.max_terms_per_order,
            )
        )

    return InteractionSelectionResult(
        selected_terms=tuple(selected),
        scores=tuple(all_scores),
        candidates_by_order=candidates_by_order,
        thresholds_by_order=thresholds_by_order,
        feature_names=feature_groups.names,
    )


__all__ = ["InteractionSearchConfig", "select_interactions"]
