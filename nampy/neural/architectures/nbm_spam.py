"""Combined Neural Basis Model and SPAM architecture."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from ..configs.nbm_spam_config import DefaultNBMSPAMConfig
from .components.base_model import BaseModel
from .components.feature_metadata import ordered_feature_keys
from .nbm import (
    _atomic_feature_names,
    _concatenate_features,
    _NBMCore,
    _normalize_nary,
)
from .spam import _SPAMCore


class NBMSPAM(BaseModel):
    """Learn unary NBM scores and combine them with low-rank polynomials."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNBMSPAMConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = DefaultNBMSPAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self._validate_features(num_feature_info, cat_feature_info)

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.output_penalty = float(
            self.hparams.get("output_penalty", config.output_penalty)
        )
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
        num_concepts = len(self.atomic_feature_names)
        nary = _normalize_nary(
            nary=self.hparams.get("nary"),
            interaction_degree=self.hparams.get("interaction_degree"),
            order=1,
            num_concepts=num_concepts,
        )
        if list(nary) != ["1"]:
            raise ValueError("NBM-SPAM requires unary NBM terms only.")

        self.ranks = [int(value) for value in self.hparams.get("ranks", config.ranks)]
        if not self.ranks or any(value <= 0 for value in self.ranks):
            raise ValueError("NBM-SPAM ranks must be a non-empty positive list.")
        self.num_subnets_per_polynomial = int(
            self.hparams.get("num_subnets", config.num_subnets)
        )
        effective_subnets = (len(self.ranks) + 1) * self.num_subnets_per_polynomial
        self.core = _NBMCore(
            nary=nary,
            num_bases=self.hparams.get("num_bases", config.num_bases),
            num_subnets=effective_subnets,
            layer_sizes=self.hparams.get("layer_sizes", config.layer_sizes),
            activation=self.hparams.get("activation", config.activation),
            dropout=self.hparams.get("dropout", config.dropout),
            bases_dropout=self.hparams.get("bases_dropout", config.bases_dropout),
            norm=self.hparams.get("norm", config.norm),
            use_glu=self.hparams.get("use_glu", config.use_glu),
            skip_connections=self.hparams.get(
                "skip_connections", config.skip_connections
            ),
            batch_norm=self.hparams.get("batch_norm", config.batch_norm),
            layer_norm=self.hparams.get("layer_norm", config.layer_norm),
            featurizer=self.hparams.get("featurizer", config.featurizer),
            sparse=False,
            nary_ignore_input=0.0,
        )

        self.segment_width = num_concepts * self.num_subnets_per_polynomial
        self.linear_head = nn.Linear(self.segment_width, int(num_classes), bias=True)
        self.polynomial_heads = nn.ModuleList()
        for index, rank in enumerate(self.ranks):
            degree_ranks = [0] * index + [rank]
            self.polynomial_heads.append(
                _SPAMCore(
                    num_concepts=self.segment_width,
                    num_outputs=int(num_classes),
                    ranks=degree_ranks,
                    dropout=self.hparams.get("spam_dropout", config.spam_dropout),
                    ignore_unary=True,
                    reg_order=self.hparams.get("reg_order", config.reg_order),
                    lower_order_correction=self.hparams.get(
                        "lower_order_correction", config.lower_order_correction
                    ),
                    use_geometric_mean=False,
                    orthogonal=self.hparams.get("orthogonal", config.orthogonal),
                    proximal=self.hparams.get("proximal", config.proximal),
                    intercept=True,
                )
            )

        self.linear_term_channels: OrderedDict[str, list[int]] = OrderedDict()
        for subnet in range(self.num_subnets_per_polynomial):
            for term_index, feature_name in enumerate(self.atomic_feature_names):
                channel = subnet * num_concepts + term_index
                self.linear_term_channels.setdefault(feature_name, []).append(channel)

    @property
    def intercept(self) -> torch.Tensor:
        result = self.linear_head.bias
        assert result is not None
        for head in self.polynomial_heads:
            assert head.classifier.bias is not None
            result = result + head.classifier.bias
        return result

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        inputs = _concatenate_features(
            num_features,
            cat_features,
            self.num_feature_keys,
            self.cat_feature_keys,
            self.feature_order,
        )
        scores = self.core(inputs)
        segments = [
            scores[:, start : start + self.segment_width]
            for start in range(0, scores.shape[1], self.segment_width)
        ]
        linear_output = self.linear_head(segments[0])
        polynomial_outputs = [
            head(segment)
            for head, segment in zip(
                self.polynomial_heads, segments[1:], strict=True
            )
        ]
        output = linear_output
        for polynomial_output in polynomial_outputs:
            output = output + polynomial_output

        result = {"output": output}
        linear_contributions = segments[0].unsqueeze(-1) * (
            self.linear_head.weight.T.unsqueeze(0)
        )
        for feature_name, channel_indices in self.linear_term_channels.items():
            result[feature_name] = linear_contributions[:, channel_indices].sum(dim=1)
        for index, (head, polynomial_output) in enumerate(
            zip(self.polynomial_heads, polynomial_outputs, strict=True)
        ):
            degree = index + 2
            bias = head.classifier.bias
            assert bias is not None
            result[f"degree_{degree}"] = polynomial_output - bias
        result["intercept"] = self.intercept

        if self.training and self.output_penalty > 0:
            result["output_penalty"] = self.output_penalty * scores.square().mean()
        if self.training and self.regularization_scale > 0:
            penalty = sum(
                (head.tensor_regularization() for head in self.polynomial_heads),
                start=output.new_zeros(()),
            )
            result["tensor_regularizer"] = self.regularization_scale * penalty
        if self.training and self.basis_l1_scale > 0:
            penalty = sum(
                (head.basis_l1_regularization() for head in self.polynomial_heads),
                start=output.new_zeros(()),
            )
            result["basis_l1_regularizer"] = self.basis_l1_scale * penalty
        return result


__all__ = ["NBMSPAM"]
