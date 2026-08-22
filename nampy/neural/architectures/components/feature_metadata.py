"""Helpers for consuming pristine PreTab's block-level feature metadata."""

from __future__ import annotations

from typing import Mapping


def ordered_feature_keys(
    num_feature_info: Mapping[str, Mapping],
    cat_feature_info: Mapping[str, Mapping],
) -> list[tuple[str, str]]:
    """Return blocks in pristine PreTab's numerical-then-categorical order."""
    return [
        *(("num", key) for key in num_feature_info),
        *(("cat", key) for key in cat_feature_info),
    ]
