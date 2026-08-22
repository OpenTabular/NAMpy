"""Single source of truth for the module forward-output dict contract.

Neural architectures return a dict from ``forward``: an ``"output"`` tensor, optional
per-feature contribution keys, ``"f1:f2"`` interaction keys, an optional
``"intercept"``, and scalar loss terms whose keys end in one of
``PENALTY_SUFFIXES``. The trainer and every module must agree on that key
grammar; this module owns it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

import torch

PENALTY_SUFFIXES = ("_penalty", "_regularizer")
RESERVED_RESULT_KEYS = frozenset({"output", "intercept"})


@runtime_checkable
class FixedLinearDesignProvider(Protocol):
    """Optional architecture contract for exact fixed-design regression."""

    supports_fixed_linear_regression: bool
    solver: str
    ridge: float
    cg_rtol: float
    cg_max_iter: int | None
    intercept: torch.Tensor | None

    def linear_design(
        self,
        num_features: Mapping[str, torch.Tensor],
        cat_features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...

    def set_linear_coefficients(
        self, coefficients: torch.Tensor, intercept: torch.Tensor | None = None
    ) -> None: ...


@runtime_checkable
class NativeTrainingProvider(Protocol):
    """Optional contract for architectures with a non-Lightning fit algorithm.

    Native trainers still consume NAMpy's train-only preprocessing and shared
    objective semantics. They own only the architecture-specific optimization
    procedure and return serializable diagnostics to the estimator facade.
    """

    supports_native_training: bool

    def fit_native(
        self,
        *,
        train_num_features: Mapping[str, torch.Tensor],
        train_cat_features: Mapping[str, torch.Tensor],
        train_targets: torch.Tensor,
        val_num_features: Mapping[str, torch.Tensor],
        val_cat_features: Mapping[str, torch.Tensor],
        val_targets: torch.Tensor,
        objective: object,
        train_offset: torch.Tensor | None = None,
        val_offset: torch.Tensor | None = None,
        train_sample_weight: torch.Tensor | None = None,
        val_sample_weight: torch.Tensor | None = None,
        random_state: int = 0,
    ) -> Mapping[str, object]: ...


def is_penalty_key(key: str) -> bool:
    return key.endswith(PENALTY_SUFFIXES)


def harvest_penalties(result: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Sum every penalty/regularizer entry of a forward-output dict."""
    total: torch.Tensor | None = None
    for key, value in result.items():
        if is_penalty_key(key):
            total = value if total is None else total + value
    return total


def validate_feature_names(
    feature_names: Iterable[str],
    *,
    owner: str,
    extra_reserved: Iterable[str] = (),
) -> None:
    """Reject feature names that collide with the output-dict key grammar."""
    names = set(feature_names)
    invalid = (RESERVED_RESULT_KEYS | set(extra_reserved)) & names
    invalid |= {name for name in names if is_penalty_key(name)}
    invalid |= {name for name in names if ":" in name}
    if invalid:
        raise ValueError(
            f"Feature names {sorted(invalid)} are reserved by {owner}: they "
            "collide with output-dict keys ('output', 'intercept', "
            "interaction names containing ':', or the "
            f"{'/'.join(PENALTY_SUFFIXES)} penalty suffixes)."
        )
