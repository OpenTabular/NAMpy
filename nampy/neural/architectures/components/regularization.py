"""Shared regularizers for models exposing additive term outputs."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def evaluating(modules: Iterable[nn.Module]):
    """Temporarily use evaluation behavior while retaining autograd."""
    states = {}
    for root in modules:
        for module in root.modules():
            states[module] = module.training
            module.training = False
    try:
        yield
    finally:
        for module, training in states.items():
            module.training = training


def mean_squared_term_outputs(outputs: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return the mean of the per-term mean squared contributions."""
    terms = list(outputs)
    if not terms:
        raise ValueError("At least one term output is required for regularization.")
    return torch.stack([torch.mean(output.square()) for output in terms]).mean()


def normalized_parameter_l2(
    module: nn.Module,
    *,
    normalizer: int | float = 1,
    half: bool = True,
) -> torch.Tensor:
    """Return a normalized L2 parameter penalty with optional ``1/2`` factor."""
    if normalizer <= 0:
        raise ValueError("normalizer must be positive.")
    parameters = list(module.parameters())
    if not parameters:
        raise ValueError("Cannot regularize a module without parameters.")
    penalty = torch.stack([parameter.square().sum() for parameter in parameters]).sum()
    if half:
        penalty = 0.5 * penalty
    return penalty / float(normalizer)
