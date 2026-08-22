"""Archipelago finite-difference interaction scores."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import torch

from .contracts import (
    FeatureGroups,
    Interaction,
    InteractionScore,
    PredictionFunction,
)


class ArchipelagoDetector:
    """Estimate arbitrary-order interactions from inclusion/removal contrasts.

    ``baseline="pairwise"`` mirrors SIAN's triangle-marginal aggregation by
    contrasting pairs of observed rows. ``baseline="zero"`` contrasts sampled
    rows against a zero vector on the transformed scale.
    """

    def __init__(
        self,
        *,
        baseline: str = "pairwise",
        max_samples: int = 128,
        max_pairs: int = 1024,
        batch_size: int = 1024,
        output_index: int = 0,
        random_state: int = 0,
    ):
        if baseline not in {"pairwise", "zero"}:
            raise ValueError("baseline must be 'pairwise' or 'zero'.")
        if max_samples < 1 or max_pairs < 1 or batch_size < 1:
            raise ValueError("Sampling and batch-size limits must be positive.")
        if output_index < 0:
            raise ValueError("output_index must be non-negative.")
        self.baseline = baseline
        self.max_samples = int(max_samples)
        self.max_pairs = int(max_pairs)
        self.batch_size = int(batch_size)
        self.output_index = int(output_index)
        self.random_state = int(random_state)

    def _project(self, predictions: torch.Tensor) -> torch.Tensor:
        if predictions.ndim == 1:
            if self.output_index != 0:
                raise ValueError("A one-dimensional predictor only has output_index=0.")
            return predictions
        flat = predictions.reshape(predictions.shape[0], -1)
        if self.output_index >= flat.shape[1]:
            raise ValueError(
                f"output_index={self.output_index} exceeds predictor width "
                f"{flat.shape[1]}."
            )
        return flat[:, self.output_index]

    def _predict(self, predict: PredictionFunction, rows: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for start in range(0, rows.shape[0], self.batch_size):
                outputs.append(self._project(predict(rows[start : start + self.batch_size])))
        return torch.cat(outputs)

    def _contrasts(self, inputs: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_samples = min(int(inputs.shape[0]), self.max_samples)
        if n_samples == 0:
            raise ValueError("Archipelago requires at least one input row.")
        sampled = inputs[:n_samples]
        if self.baseline == "zero":
            zero = torch.zeros_like(sampled[0])
            return [(row, zero) for row in sampled[: self.max_pairs]]

        pairs = list(combinations(range(n_samples), 2))
        if not pairs:
            return [(sampled[0], torch.zeros_like(sampled[0]))]
        if len(pairs) > self.max_pairs:
            generator = torch.Generator().manual_seed(self.random_state)
            order = torch.randperm(len(pairs), generator=generator)[: self.max_pairs]
            pairs = [pairs[int(index)] for index in order]
        return [(sampled[left], sampled[right]) for left, right in pairs]

    @staticmethod
    def _hybrid(
        context: torch.Tensor,
        insertion: torch.Tensor,
        subset: Sequence[str],
        feature_groups: FeatureGroups,
    ) -> torch.Tensor:
        row = context.clone()
        indices = feature_groups.indices(subset)
        row[list(indices)] = insertion[list(indices)]
        return row

    def _candidate_score(
        self,
        predict: PredictionFunction,
        contrasts: Sequence[tuple[torch.Tensor, torch.Tensor]],
        interaction: Interaction,
        feature_groups: FeatureGroups,
    ) -> InteractionScore:
        subsets = [
            tuple(interaction[index] for index in range(len(interaction)) if mask & (1 << index))
            for mask in range(1 << len(interaction))
        ]
        rows = []
        for target, baseline in contrasts:
            rows.extend(
                self._hybrid(baseline, target, subset, feature_groups)
                for subset in subsets
            )
            rows.extend(
                self._hybrid(target, baseline, subset, feature_groups)
                for subset in subsets
            )
        values = self._predict(predict, torch.stack(rows)).reshape(
            len(contrasts), 2, len(subsets)
        )

        inclusion_values = []
        removal_values = []
        total_values = []
        order = len(interaction)
        for contrast_values in values:
            inclusion = contrast_values[0]
            removal = contrast_values[1]
            inclusion_score = sum(
                ((-1) ** (order - len(subset))) * inclusion[index]
                for index, subset in enumerate(subsets)
            )
            removal_score = sum(
                ((-1) ** len(subset)) * removal[index]
                for index, subset in enumerate(subsets)
            )
            inclusion_values.append(inclusion_score)
            removal_values.append(removal_score)
            total_values.append(
                0.5
                * (
                    torch.abs(inclusion[-1] - inclusion[0])
                    + torch.abs(removal[0] - removal[-1])
                )
            )

        inclusion_tensor = torch.stack(inclusion_values)
        removal_tensor = torch.stack(removal_values)
        score = 0.5 * (inclusion_tensor.square() + removal_tensor.square())
        signed = 0.5 * (inclusion_tensor + removal_tensor)
        return InteractionScore(
            interaction=interaction,
            score=float(score.mean().cpu()),
            inclusion=float(inclusion_tensor.mean().cpu()),
            removal=float(removal_tensor.mean().cpu()),
            signed_score=float(signed.mean().cpu()),
            total_effect=float(torch.stack(total_values).mean().cpu()),
            n_contrasts=len(contrasts),
        )

    def score(
        self,
        predict: PredictionFunction,
        inputs: torch.Tensor,
        candidates: Sequence[Interaction],
        feature_groups: FeatureGroups,
    ) -> list[InteractionScore]:
        if inputs.ndim != 2:
            raise ValueError("Archipelago inputs must have shape [rows, columns].")
        contrasts = self._contrasts(inputs)
        return [
            self._candidate_score(predict, contrasts, tuple(candidate), feature_groups)
            for candidate in candidates
        ]


__all__ = ["ArchipelagoDetector"]
