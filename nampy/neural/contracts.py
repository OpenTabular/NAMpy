"""Single source of truth for the module forward-output dict contract.

Neural modules return a dict from ``forward``: an ``"output"`` tensor, optional
per-feature contribution keys, ``"f1:f2"`` interaction keys, an optional
``"intercept"``, and scalar loss terms whose keys end in one of
``PENALTY_SUFFIXES``. The trainer and every module must agree on that key
grammar; this module owns it.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

PENALTY_SUFFIXES = ("_penalty", "_regularizer")
RESERVED_RESULT_KEYS = frozenset({"output", "intercept"})


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

