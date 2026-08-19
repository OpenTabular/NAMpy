"""Shared interaction-term scaffolding for additive architectures.

NAM, QNAM, TreeNAM, and SplineNAM enumerate feature interactions the same
way — ``itertools.combinations`` over the feature names with ``":"``-joined
keys (the key grammar specified in ``nampy/neural/contracts.py``) — and run
the same gather loop in ``forward``. The architecture-specific pieces (how a
subnetwork is built for an interaction, and how its raw output is produced or
post-processed) stay with each model and are passed in as callables.

NATT and NAMformer deliberately keep their own interaction code: their
``feature_dropout`` is a plain elementwise ``nn.Dropout`` over concatenated
term outputs, not the term-mask dropout in :func:`apply_feature_dropout`, and
NAMformer feeds transformer embeddings rather than raw feature tensors.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn


def sum_feature_dims(
    interaction: Sequence[str],
    num_feature_info: Mapping[str, Mapping],
    cat_feature_info: Mapping[str, Mapping],
) -> int:
    """Input dimension of an interaction: sum of the member feature dimensions."""
    input_dim = 0
    for feature in interaction:
        if feature in num_feature_info:
            input_dim += num_feature_info[feature]["dimension"]
        elif feature in cat_feature_info:
            input_dim += cat_feature_info[feature]["dimension"]
    return input_dim


def create_interaction_networks(
    feature_names: Sequence[str],
    interaction_degree: int | None,
    make_subnetwork: Callable[[tuple[str, ...]], nn.Module],
) -> nn.ModuleDict:
    """Enumerate interactions up to ``interaction_degree`` into a ModuleDict.

    Keys follow the ``":"``-joined grammar from ``nampy/neural/contracts.py``
    (e.g. ``"x:z"``). ``make_subnetwork`` receives the tuple of interacting
    feature names and returns the module; input-dimension rules stay with the
    caller.
    """
    networks = nn.ModuleDict()
    if interaction_degree is None or interaction_degree < 2:
        return networks
    for degree in range(2, interaction_degree + 1):
        for interaction in combinations(feature_names, degree):
            interaction_name = ":".join(interaction)
            networks[interaction_name] = make_subnetwork(interaction)
    return networks


def interaction_forward(
    networks: nn.ModuleDict,
    interaction_degree: int | None,
    num_features: Mapping[str, torch.Tensor],
    cat_features: Mapping[str, torch.Tensor],
    apply_network: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Shared interaction gather loop.

    For each interaction network, concatenates the member feature tensors
    (cast to float) and calls ``apply_network(network, input_features)``.
    Model-specific behavior (extra input reduction, monotone transforms,
    penalty accumulation) lives in ``apply_network``.
    """
    interaction_outputs: dict[str, torch.Tensor] = {}
    if interaction_degree is not None and interaction_degree >= 2:
        all_features = {**num_features, **cat_features}
        for interaction_name, interaction_network in networks.items():
            feature_names = interaction_name.split(":")
            input_features = torch.cat(
                [all_features[fn] for fn in feature_names], dim=-1
            ).float()
            interaction_outputs[interaction_name] = apply_network(
                interaction_network, input_features
            )
    return interaction_outputs


def apply_feature_dropout(
    term_outputs: torch.Tensor, p: float, training: bool
) -> torch.Tensor:
    """Term-mask feature dropout on a ``[batch, terms, outputs]`` tensor.

    Draws one Bernoulli mask entry per (sample, term) and applies it across
    the whole output dimension, so entire term contributions are dropped
    consistently (preserving e.g. QNAM's within-term monotonicity).
    """
    if p > 0.0 and training:
        mask = torch.ones(
            term_outputs.shape[0],
            term_outputs.shape[1],
            1,
            device=term_outputs.device,
            dtype=term_outputs.dtype,
        )
        mask = nn.functional.dropout(mask, p=p, training=True)
        term_outputs = term_outputs * mask
    return term_outputs
