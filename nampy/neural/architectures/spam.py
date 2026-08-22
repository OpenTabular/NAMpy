"""Scalable Polynomial Additive Model (SPAM)."""

from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.spam_config import DefaultSPAMConfig
from .components.base_model import BaseModel
from .components.feature_metadata import ordered_feature_keys
from .nbm import _atomic_feature_names, _concatenate_features


class _SPAMCore(nn.Module):
    """Low-rank homogeneous polynomial features and their output head."""

    def __init__(
        self,
        *,
        num_concepts: int,
        num_outputs: int,
        ranks,
        dropout: float,
        ignore_unary: bool,
        reg_order: int,
        lower_order_correction: bool,
        use_geometric_mean: bool,
        orthogonal: bool,
        proximal: bool,
        intercept: bool,
    ) -> None:
        super().__init__()
        ranks = [int(rank) for rank in ranks]
        if not ranks:
            raise ValueError("SPAM ranks must contain at least one degree.")
        if any(rank < 0 for rank in ranks):
            raise ValueError("SPAM ranks must be non-negative.")
        if int(reg_order) < 1:
            raise ValueError("reg_order must be positive.")

        self.num_concepts = int(num_concepts)
        self.num_outputs = int(num_outputs)
        self.ranks = ranks
        self.ignore_unary = bool(ignore_unary)
        self.reg_order = int(reg_order)
        self.lower_order_correction = bool(lower_order_correction)
        self.use_geometric_mean = bool(use_geometric_mean)
        self.proximal = bool(proximal)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.poly_weights = nn.ModuleList()
        self.degrees: list[int] = []
        num_features = 0 if self.ignore_unary else self.num_concepts
        for index, rank in enumerate(self.ranks):
            if rank == 0:
                continue
            projection: nn.Module = nn.Linear(self.num_concepts, rank, bias=False)
            if orthogonal:
                projection = nn.utils.parametrizations.orthogonal(projection)
            self.poly_weights.append(projection)
            self.degrees.append(index + 2)
            num_features += rank
        if num_features == 0:
            raise ValueError("SPAM requires unary features or at least one positive rank.")
        self.num_features = num_features
        self.classifier = nn.Linear(num_features, self.num_outputs, bias=intercept)

    def _proximal_step(self) -> None:
        with torch.no_grad():
            self.classifier.weight.clamp_(min=0)
            for projection in self.poly_weights:
                projection.weight.clamp_(min=0)

    @staticmethod
    def _signed_root(inputs: torch.Tensor, degree: int) -> torch.Tensor:
        return torch.sign(inputs) * torch.abs(inputs).pow(1.0 / degree)

    @staticmethod
    def _compute_correction(
        inputs: torch.Tensor, weight: torch.Tensor, degree: int
    ) -> torch.Tensor:
        correction = torch.zeros(
            inputs.shape[0], weight.shape[0], device=inputs.device, dtype=inputs.dtype
        )
        for exponent in range(2, degree + 1):
            coefficient = math.comb(degree, exponent) * (
                2 * (1 - (exponent % 2)) - 1
            )
            complement = degree - exponent
            first = F.linear(inputs.pow(exponent), weight.pow(exponent))
            second = 1.0
            if complement > 0:
                second = F.linear(inputs.pow(complement), weight.pow(complement))
            correction = correction + coefficient * first * second
        return correction

    def feature_blocks(self, inputs: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        blocks: list[tuple[str, torch.Tensor]] = []
        if not self.ignore_unary:
            blocks.append(("unary", inputs))
        for degree, projection in zip(
            self.degrees, self.poly_weights, strict=True
        ):
            transformed = (
                self._signed_root(inputs, degree)
                if self.use_geometric_mean
                else inputs
            )
            features = projection(transformed).pow(degree)
            if self.lower_order_correction:
                features = features - self._compute_correction(
                    transformed, projection.weight, degree
                )
            blocks.append((f"degree_{degree}", features))
        return blocks

    def forward_with_blocks(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        if self.proximal:
            self._proximal_step()
        raw_blocks = self.feature_blocks(inputs)
        features = self.dropout(
            torch.cat([values for _, values in raw_blocks], dim=1)
        )
        blocks = []
        offset = 0
        for name, values in raw_blocks:
            width = values.shape[1]
            blocks.append((name, features[:, offset : offset + width]))
            offset += width
        return self.classifier(features), blocks

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_blocks(inputs)[0]

    def tensor_regularization(self) -> torch.Tensor:
        loss = torch.linalg.vector_norm(
            self.classifier.weight, ord=self.reg_order, dim=0
        ).sum()
        if self.classifier.bias is not None:
            loss = loss + torch.linalg.vector_norm(
                self.classifier.bias, ord=self.reg_order
            )
        for degree, projection in zip(
            self.degrees, self.poly_weights, strict=True
        ):
            loss = loss + torch.linalg.vector_norm(
                projection.weight, ord=self.reg_order, dim=0
            ).pow(degree).sum()
        return loss

    def basis_l1_regularization(self) -> torch.Tensor:
        if not self.poly_weights:
            return self.classifier.weight.new_zeros(())
        return torch.stack(
            [projection.weight.abs().sum() for projection in self.poly_weights]
        ).sum()

    def local_term_importance(
        self,
        inputs: torch.Tensor,
        *,
        target: int = 0,
        top_k: int = 10,
    ) -> list[list[tuple[tuple[int, ...], float]]]:
        """Return upstream-style local unary/polynomial term importances.

        The released SPAM explanation expands distinct-variable monomials and
        folds each polynomial's diagonal terms into the unary effects.  This
        method preserves that definition while returning semantic index tuples
        instead of the upstream flattened feature positions.
        """
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [batch, concepts].")
        if target < 0 or target >= self.num_outputs:
            raise ValueError("target is outside the SPAM output range.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if self.ignore_unary:
            raise ValueError(
                "Local source-term importance requires ignore_unary=False."
            )

        classifier_weights = self.classifier.weight[target]
        rows = []
        for row in inputs:
            nonzero = torch.nonzero(row, as_tuple=False).view(-1).tolist()
            unary = {
                (index,): float(
                    (row[index] * classifier_weights[index]).detach().item()
                )
                for index in nonzero
            }
            classifier_offset = self.num_concepts
            candidates: list[tuple[tuple[int, ...], float]] = []
            for degree, projection in zip(
                self.degrees, self.poly_weights, strict=True
            ):
                transformed = (
                    self._signed_root(row, degree)
                    if self.use_geometric_mean
                    else row
                )
                rank = projection.weight.shape[0]
                degree_weights = classifier_weights[
                    classifier_offset : classifier_offset + rank
                ]
                for index in nonzero:
                    diagonal = torch.sum(
                        degree_weights
                        * (projection.weight[:, index] * transformed[index]).pow(
                            degree
                        )
                    )
                    unary[(index,)] += float(diagonal.detach().item())
                activations = projection.weight * transformed.unsqueeze(0)
                for term in combinations(nonzero, degree):
                    contribution = torch.sum(
                        degree_weights
                        * torch.prod(activations[:, list(term)], dim=1)
                    )
                    candidates.append(
                        (tuple(term), float(contribution.detach().item()))
                    )
                classifier_offset += rank
            candidates.extend(unary.items())
            candidates.sort(key=lambda item: abs(item[1]), reverse=True)
            rows.append(candidates[:top_k])
        return rows


class SPAM(BaseModel):
    """Standalone SPAM architecture with degree-wise additive outputs."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultSPAMConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = DefaultSPAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self._validate_features(num_feature_info, cat_feature_info)

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.regularization_scale = float(
            self.hparams.get("regularization_scale", config.regularization_scale)
        )
        self.basis_l1_scale = float(
            self.hparams.get(
                "basis_l1_regularization", config.basis_l1_regularization
            )
        )

        self.num_feature_keys = list(num_feature_info)
        self.cat_feature_keys = list(cat_feature_info)
        self.feature_order = ordered_feature_keys(
            num_feature_info, cat_feature_info
        )
        self.atomic_feature_names = _atomic_feature_names(
            num_feature_info, cat_feature_info
        )
        self.core = _SPAMCore(
            num_concepts=len(self.atomic_feature_names),
            num_outputs=int(num_classes),
            ranks=self.hparams.get("ranks", config.ranks),
            dropout=self.hparams.get("dropout", config.dropout),
            ignore_unary=self.hparams.get("ignore_unary", config.ignore_unary),
            reg_order=self.hparams.get("reg_order", config.reg_order),
            lower_order_correction=self.hparams.get(
                "lower_order_correction", config.lower_order_correction
            ),
            use_geometric_mean=self.hparams.get(
                "use_geometric_mean", config.use_geometric_mean
            ),
            orthogonal=self.hparams.get("orthogonal", config.orthogonal),
            proximal=self.hparams.get("proximal", config.proximal),
            intercept=self.hparams.get("intercept", config.intercept),
        )

    @property
    def intercept(self) -> nn.Parameter | None:
        return self.core.classifier.bias

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        inputs = _concatenate_features(
            num_features,
            cat_features,
            self.num_feature_keys,
            self.cat_feature_keys,
            self.feature_order,
        )
        output, blocks = self.core.forward_with_blocks(inputs)
        result = {"output": output}
        offset = 0
        for block_name, block in blocks:
            width = block.shape[1]
            weights = self.core.classifier.weight[:, offset : offset + width]
            contributions = block.unsqueeze(-1) * weights.T.unsqueeze(0)
            if block_name == "unary":
                for index, feature_name in enumerate(self.atomic_feature_names):
                    result[feature_name] = contributions[:, index]
            else:
                result[block_name] = contributions.sum(dim=1)
            offset += width
        if self.core.classifier.bias is not None:
            result["intercept"] = self.core.classifier.bias
        if self.training and self.regularization_scale > 0:
            result["tensor_regularizer"] = (
                self.regularization_scale * self.core.tensor_regularization()
            )
        if self.training and self.basis_l1_scale > 0:
            result["basis_l1_regularizer"] = (
                self.basis_l1_scale * self.core.basis_l1_regularization()
            )
        return result

    def local_term_importance(
        self,
        num_features: dict,
        cat_features: dict,
        *,
        target: int = 0,
        top_k: int = 10,
    ) -> list[list[dict[str, object]]]:
        """Return local top-k source terms using SPAM's polynomial expansion."""
        inputs = _concatenate_features(
            num_features,
            cat_features,
            self.num_feature_keys,
            self.cat_feature_keys,
            self.feature_order,
        )
        rows = self.core.local_term_importance(
            inputs, target=target, top_k=top_k
        )
        return [
            [
                {
                    "term": tuple(self.atomic_feature_names[index] for index in term),
                    "order": len(term),
                    "contribution": contribution,
                }
                for term, contribution in row
            ]
            for row in rows
        ]


__all__ = ["SPAM"]
