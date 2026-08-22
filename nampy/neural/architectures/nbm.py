"""Neural Basis Model architecture.

The dense and sparse execution paths mirror ``ConceptNBMNary`` and
``ConceptNBMNarySparse`` from the NBM-SPAM reference repository. NAMpy's
additional basis-network controls remain ordinary configuration options; the
defaults match the released dense NBM definition.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from itertools import combinations
from numbers import Real
from typing import Mapping

import torch
import torch.nn as nn

from ..configs.nbm_config import DefaultNBMConfig
from .components.base_model import BaseModel
from .components.concept_bases import ConceptNNBasesNary
from .components.feature_metadata import ordered_feature_keys


def _atomic_feature_names(
    num_feature_info: Mapping[str, Mapping],
    cat_feature_info: Mapping[str, Mapping],
) -> list[str]:
    names: list[str] = []
    feature_info = {"num": num_feature_info, "cat": cat_feature_info}
    for kind, feature_name in ordered_feature_keys(
        num_feature_info, cat_feature_info
    ):
        info = feature_info[kind][feature_name]
        dimension = int(info["dimension"])
        if dimension <= 0:
            raise ValueError(
                f"Feature {feature_name!r} has invalid dimension {dimension}."
            )
        if dimension == 1:
            names.append(feature_name)
        else:
            names.extend(f"{feature_name}[{index}]" for index in range(dimension))
    return names


def _concatenate_features(
    num_features: Mapping[str, torch.Tensor],
    cat_features: Mapping[str, torch.Tensor],
    num_feature_keys: list[str],
    cat_feature_keys: list[str],
    feature_order: list[tuple[str, str]] | None = None,
) -> torch.Tensor:
    tensors = []
    order = feature_order or [
        *(("num", key) for key in num_feature_keys),
        *(("cat", key) for key in cat_feature_keys),
    ]
    for kind, feature_name in order:
        source = num_features if kind == "num" else cat_features
        tensor = source[feature_name]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        tensors.append(tensor)
    if not tensors:
        raise ValueError("NBM received no input features.")
    return torch.cat(tensors, dim=1).float()


def _normalize_nary(
    *, nary, interaction_degree, order, num_concepts: int
) -> OrderedDict[str, list[tuple[int, ...]]]:
    if nary is not None:
        if isinstance(nary, (list, tuple)):
            orders = sorted({int(value) for value in nary})
            if not orders:
                raise ValueError("nary list/tuple must not be empty.")
            resolved = OrderedDict(
                (str(value), list(combinations(range(num_concepts), value)))
                for value in orders
            )
        elif isinstance(nary, dict):
            resolved = OrderedDict(
                (str(int(key)), [tuple(map(int, term)) for term in terms])
                for key, terms in nary.items()
            )
        else:
            raise TypeError("nary must be None, a list/tuple of orders, or a dict.")
    else:
        if interaction_degree is not None:
            if int(interaction_degree) < 1:
                raise ValueError("interaction_degree must be >= 1.")
            orders = range(1, int(interaction_degree) + 1)
        else:
            if int(order) < 1:
                raise ValueError("order must be >= 1.")
            orders = [int(order)]
        resolved = OrderedDict(
            (str(value), list(combinations(range(num_concepts), value)))
            for value in orders
        )

    seen: set[tuple[int, ...]] = set()
    for order_key, terms in resolved.items():
        term_order = int(order_key)
        if term_order < 1 or term_order > num_concepts:
            raise ValueError(
                f"Interaction order {term_order} must lie in [1, {num_concepts}]."
            )
        for term in terms:
            if len(term) != term_order:
                raise ValueError(
                    f"Tuple {term} does not match declared order {term_order}."
                )
            if len(set(term)) != len(term):
                raise ValueError(f"Tuple {term} repeats a concept index.")
            if any(index < 0 or index >= num_concepts for index in term):
                raise ValueError(
                    f"Tuple {term} contains an index outside [0, {num_concepts - 1}]."
                )
            canonical = tuple(sorted(term))
            if canonical in seen:
                raise ValueError(f"Duplicate NBM term {term}.")
            seen.add(canonical)
    if not seen:
        raise ValueError("NBM requires at least one unary or interaction term.")
    return resolved


class _NBMCore(nn.Module):
    """Shared-basis feature extractor used by NBM and NBM-SPAM."""

    def __init__(
        self,
        *,
        nary: OrderedDict[str, list[tuple[int, ...]]],
        num_bases: int,
        num_subnets: int,
        layer_sizes,
        activation,
        dropout: float,
        bases_dropout: float,
        norm,
        use_glu: bool,
        skip_connections: bool,
        batch_norm: bool,
        layer_norm: bool,
        featurizer: str,
        sparse: bool,
        nary_ignore_input,
    ) -> None:
        super().__init__()
        if int(num_bases) < 1 or int(num_subnets) < 1:
            raise ValueError("num_bases and num_subnets must be positive.")
        if sparse and int(num_subnets) != 1:
            raise ValueError("Sparse NBM supports num_subnets=1, matching upstream.")
        featurizer = str(featurizer).lower()
        if featurizer not in {"conv1d", "einsum"}:
            raise ValueError("featurizer must be 'conv1d' or 'einsum'.")

        self.nary = nary
        self.order_keys = list(nary)
        self.num_bases = int(num_bases)
        self.num_subnets = int(num_subnets)
        self.sparse = bool(sparse)
        self.featurizer_implementation = featurizer
        self.bases_dropout = nn.Dropout(float(bases_dropout))

        self.channel_specs: list[dict[str, object]] = []
        for order_key in self.order_keys:
            for subnet in range(self.num_subnets):
                for term in self.nary[order_key]:
                    self.channel_specs.append(
                        {
                            "order": int(order_key),
                            "subnet": subnet,
                            "tuple": tuple(term),
                        }
                    )
        self.num_channels = len(self.channel_specs)

        self._nary_buffer_names: dict[str, str] = {}
        for order_key in self.order_keys:
            name = f"_nary_idx_ord{order_key}"
            self.register_buffer(
                name,
                torch.tensor(self.nary[order_key], dtype=torch.long),
                persistent=False,
            )
            self._nary_buffer_names[order_key] = name

        self.bases_nary_models = nn.ModuleDict()
        for order_key in self.order_keys:
            subnet_count = 1 if self.sparse else self.num_subnets
            for subnet in range(subnet_count):
                self.bases_nary_models[self.get_key(order_key, subnet)] = (
                    ConceptNNBasesNary(
                        order=int(order_key),
                        num_bases=self.num_bases,
                        layer_sizes=list(layer_sizes),
                        activation=activation,
                        dropout=float(dropout),
                        use_batch_norm=bool(batch_norm),
                        use_layer_norm=bool(layer_norm),
                        norm=norm,
                        use_glu=bool(use_glu),
                        skip_connections=bool(skip_connections),
                    )
                )

        if self.sparse:
            self.featurizer = None
            self.register_parameter("featurizer_weight", None)
            self.register_parameter("featurizer_bias", None)
            self.featurizer_params = nn.ModuleDict()
            for order_key in self.order_keys:
                num_terms = len(self.nary[order_key])
                parameters = nn.ParameterDict(
                    {
                        "weight": nn.Parameter(torch.empty(num_terms, self.num_bases)),
                        "bias": nn.Parameter(torch.empty(num_terms)),
                    }
                )
                nn.init.kaiming_uniform_(parameters["weight"], a=math.sqrt(5))
                bound = 1 / math.sqrt(self.num_bases)
                nn.init.uniform_(parameters["bias"], -bound, bound)
                self.featurizer_params[order_key] = parameters
            if isinstance(nary_ignore_input, Real):
                self.nary_ignore_input = OrderedDict(
                    (key, float(nary_ignore_input)) for key in self.order_keys
                )
            elif isinstance(nary_ignore_input, dict):
                normalized_ignore = {
                    str(key): value for key, value in nary_ignore_input.items()
                }
                missing = set(self.order_keys) - set(normalized_ignore)
                if missing:
                    raise ValueError(
                        "nary_ignore_input is missing orders " f"{sorted(missing)}."
                    )
                self.nary_ignore_input = OrderedDict(
                    (key, float(normalized_ignore[key])) for key in self.order_keys
                )
            else:
                raise TypeError("nary_ignore_input must be a number or dictionary.")
        else:
            self.featurizer_params = None
            self.nary_ignore_input = None
            if featurizer == "conv1d":
                # Mirrors upstream ConceptNBMNary.__init__ directly.
                self.featurizer = nn.Conv1d(
                    in_channels=self.num_channels * self.num_bases,
                    out_channels=self.num_channels,
                    kernel_size=1,
                    groups=self.num_channels,
                )
                self.register_parameter("featurizer_weight", None)
                self.register_parameter("featurizer_bias", None)
            else:
                self.featurizer = None
                self.featurizer_weight = nn.Parameter(
                    torch.empty(self.num_channels, self.num_bases)
                )
                self.featurizer_bias = nn.Parameter(torch.empty(self.num_channels))
                nn.init.kaiming_uniform_(
                    self.featurizer_weight, a=math.sqrt(5)
                )
                bound = 1 / math.sqrt(self.num_bases)
                nn.init.uniform_(self.featurizer_bias, -bound, bound)

    @staticmethod
    def get_key(order, subnet) -> str:
        return f"ord{order}_net{subnet}"

    def _dense_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chunks = []
        for order_key in self.order_keys:
            indices = getattr(self, self._nary_buffer_names[order_key])
            order = indices.shape[1]
            selected = inputs[:, indices]
            flattened = selected.reshape(-1, order)
            for subnet in range(self.num_subnets):
                bases = self.bases_nary_models[self.get_key(order_key, subnet)](
                    flattened
                )
                bases = self.bases_dropout(bases)
                chunks.append(
                    bases.reshape(inputs.shape[0], indices.shape[0], self.num_bases)
                )
        all_bases = torch.cat(chunks, dim=1)
        if self.featurizer is not None:
            # Mirrors upstream ConceptNBMNary.forward.
            return self.featurizer(
                all_bases.reshape(inputs.shape[0], -1, 1)
            ).squeeze(-1)
        assert self.featurizer_weight is not None
        assert self.featurizer_bias is not None
        return (
            torch.einsum("btk,tk->bt", all_bases, self.featurizer_weight)
            + self.featurizer_bias
        )

    def _sparse_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chunks = []
        assert self.featurizer_params is not None
        assert self.nary_ignore_input is not None
        for order_key in self.order_keys:
            indices = getattr(self, self._nary_buffer_names[order_key])
            selected = inputs[:, indices]
            active = torch.any(selected != self.nary_ignore_input[order_key], dim=-1)
            dense_scores = inputs.new_zeros(selected.shape[:2])
            if torch.any(active):
                active_inputs = selected[active]
                bases = self.bases_nary_models[self.get_key(order_key, 0)](
                    active_inputs
                )
                bases = self.bases_dropout(bases)
                term_positions = active.nonzero(as_tuple=False)[:, 1]
                parameters = self.featurizer_params[order_key]
                weights = parameters["weight"][term_positions]
                biases = parameters["bias"][term_positions]
                dense_scores[active] = (weights * bases).sum(dim=-1) + biases
            chunks.append(dense_scores)
        return torch.cat(chunks, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._sparse_forward(inputs) if self.sparse else self._dense_forward(inputs)


class NBM(BaseModel):
    """Neural Basis Model with configurable n-ary and sparse execution."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNBMConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = DefaultNBMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self._validate_features(num_feature_info, cat_feature_info)

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.num_classes = int(num_classes)
        self.output_penalty = float(
            self.hparams.get("output_penalty", config.output_penalty)
        )
        self.feature_dropout_p = float(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

        self.num_feature_keys = list(num_feature_info)
        self.cat_feature_keys = list(cat_feature_info)
        self.feature_order = ordered_feature_keys(
            num_feature_info, cat_feature_info
        )
        self.atomic_feature_names = _atomic_feature_names(
            num_feature_info, cat_feature_info
        )
        self.num_atomic_features = len(self.atomic_feature_names)
        self.nary = _normalize_nary(
            nary=self.hparams.get("nary", config.nary),
            interaction_degree=self.hparams.get(
                "interaction_degree", config.interaction_degree
            ),
            order=self.hparams.get("order", config.order),
            num_concepts=self.num_atomic_features,
        )

        self.core = _NBMCore(
            nary=self.nary,
            num_bases=self.hparams.get("num_bases", config.num_bases),
            num_subnets=self.hparams.get("num_subnets", config.num_subnets),
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
            sparse=self.hparams.get("sparse", config.sparse),
            nary_ignore_input=self.hparams.get(
                "nary_ignore_input", config.nary_ignore_input
            ),
        )
        self.num_subnets = self.core.num_subnets
        self.num_bases = self.core.num_bases
        self.num_channels = self.core.num_channels
        self.sparse = self.core.sparse

        self.channel_specs: list[dict[str, object]] = []
        self.term_to_channel_indices: OrderedDict[str, list[int]] = OrderedDict()
        for channel_index, spec in enumerate(self.core.channel_specs):
            term = spec["tuple"]
            assert isinstance(term, tuple)
            name = ":".join(self.atomic_feature_names[index] for index in term)
            self.channel_specs.append(
                {
                    **spec,
                    "name": name,
                    "kind": "main" if len(term) == 1 else "interaction",
                }
            )
            self.term_to_channel_indices.setdefault(name, []).append(channel_index)

        use_intercept = bool(self.hparams.get("intercept", config.intercept))
        self.classifier = nn.Linear(
            self.num_channels, self.num_classes, bias=use_intercept
        )

    @property
    def intercept(self) -> nn.Parameter | None:
        return self.classifier.bias

    @property
    def bases_nary_models(self) -> nn.ModuleDict:
        return self.core.bases_nary_models

    @property
    def featurizer(self):
        return self.core.featurizer

    def _apply_term_dropout(self, scores: torch.Tensor) -> torch.Tensor:
        if self.feature_dropout_p <= 0 or not self.training:
            return scores
        logical_mask = scores.new_ones(
            scores.shape[0], len(self.term_to_channel_indices)
        )
        logical_mask = nn.functional.dropout(
            logical_mask, p=self.feature_dropout_p, training=True
        )
        channel_mask = torch.ones_like(scores)
        for term_index, channel_indices in enumerate(
            self.term_to_channel_indices.values()
        ):
            channel_mask[:, channel_indices] = logical_mask[
                :, term_index : term_index + 1
            ]
        return scores * channel_mask

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        inputs = _concatenate_features(
            num_features,
            cat_features,
            self.num_feature_keys,
            self.cat_feature_keys,
            self.feature_order,
        )
        reference_scores = self.core(inputs)
        scores = self._apply_term_dropout(reference_scores)
        contributions = scores.unsqueeze(-1) * self.classifier.weight.T.unsqueeze(0)
        output = contributions.sum(dim=1)
        if self.classifier.bias is not None:
            output = output + self.classifier.bias

        result = {"output": output}
        for name, channel_indices in self.term_to_channel_indices.items():
            result[name] = contributions[:, channel_indices].sum(dim=1)
        if self.classifier.bias is not None:
            result["intercept"] = self.classifier.bias
        if self.training and self.output_penalty > 0:
            result["output_penalty"] = (
                self.output_penalty * reference_scores.square().mean()
            )
        return result


__all__ = ["NBM"]
