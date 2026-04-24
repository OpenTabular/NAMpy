import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from tests._taxonomy_registry import (
    _DEFAULT_MARKS_BY_FILE,
    _FAMILY_MARK_NAMES,
    _METHOD_MARK_NAMES,
    _SELECTION_CAPABLE_FILES,
    _SMOOTH_MARK_NAMES,
    _STATUS_MARKS_BY_FILE,
)


@pytest.fixture(autouse=True)
def _set_seeds():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.uniform(-1, 1, size=40),
        }
    )
    y = (
        0.5 * X["f1"].to_numpy()
        - 0.2 * X["f2"].to_numpy()
        + rng.normal(scale=0.1, size=40)
    )
    return X, y


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.uniform(-1, 1, size=40),
        }
    )
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    return X, y


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=40),
            "num2": rng.uniform(-1, 1, size=40),
            "int_cat": rng.integers(0, 3, size=40),
            "str_cat": rng.choice(["a", "b", "c"], size=40),
        }
    )
    y = (X["num1"] * 0.3 + rng.normal(scale=0.1, size=40)).to_numpy()
    return X, y


_SURFACE_HINTS = {
    "surface_output": ("anova", "predict", "prediction", "residual", "vcov", "terms"),
    "surface_smoothcon": ("smoothcon", "basis", "penalt"),
    "surface_trace": ("trace",),
    "surface_kcheck": ("k_check",),
    "surface_derivatives": (
        "gradient",
        "hessian",
        "derivative",
        "criterion",
        "laplace",
        "fd",
    ),
    "surface_snapshot": ("snapshot", "parity_snapshot"),
    "surface_backend": ("backend",),
}


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def _extract_case_like_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    out = {}
    for attr in ("case_id", "formula", "family", "method", "select", "weights_column"):
        if hasattr(value, attr):
            out[attr] = getattr(value, attr)
    return out


def _normalize_family_name(value: Any) -> str | None:
    if isinstance(value, dict):
        name = str(value.get("name", "")).lower()
        if name in {"negativebinomial", "negative_binomial"}:
            return "negbin"
        return name or None
    if value is None:
        return None
    name = str(value).lower()
    if name in {"negativebinomial", "negative_binomial"}:
        return "negbin"
    return name


def _collect_text_fragments(item: pytest.Item) -> list[str]:
    texts = [item.nodeid.lower(), item.name.lower()]
    if item.cls is not None:
        texts.append(item.cls.__name__.lower())
    if hasattr(item, "callspec"):
        for value in item.callspec.params.values():
            texts.extend(_collect_value_strings(value))
    return texts


def _collect_value_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, dict):
        out: list[str] = []
        for key, val in value.items():
            out.append(str(key).lower())
            out.extend(_collect_value_strings(val))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for part in value:
            out.extend(_collect_value_strings(part))
        return out
    case_like = _extract_case_like_fields(value)
    if case_like:
        return _collect_value_strings(case_like)
    if isinstance(value, bool):
        return [str(value).lower()]
    return [str(value).lower()]


def _infer_marks_from_text(texts: list[str]) -> set[str]:
    joined = " ".join(texts)
    marks: set[str] = set()

    for surface_mark, hints in _SURFACE_HINTS.items():
        if any(hint in joined for hint in hints):
            marks.add(surface_mark)

    if "te(" in joined or _has_token(joined, "te"):
        marks.add("smooth_te")
    if "ti(" in joined or _has_token(joined, "ti"):
        marks.add("smooth_ti")
    if "t2(" in joined or _has_token(joined, "t2") or "tensor_anova" in joined:
        marks.add("smooth_t2")

    for token, mark in _SMOOTH_MARK_NAMES.items():
        if token in {"te", "ti", "t2"}:
            continue
        if (
            f'bs="{token}"' in joined
            or f"bs='{token}'" in joined
            or _has_token(joined, token)
        ):
            marks.add(mark)

    for family, mark in _FAMILY_MARK_NAMES.items():
        if family == "general":
            continue
        if _has_token(joined, family):
            marks.add(mark)

    for method, mark in _METHOD_MARK_NAMES.items():
        if _has_token(joined, method):
            marks.add(mark)

    if "select=true" in joined or "select true" in joined or "_select_" in joined:
        marks.add("select_true")

    return marks


def _infer_marks_from_callspec(item: pytest.Item) -> set[str]:
    marks: set[str] = set()
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return marks

    params = callspec.params
    family_value = params.get("family")
    family_name = _normalize_family_name(family_value)
    if family_name in _FAMILY_MARK_NAMES:
        marks.add(_FAMILY_MARK_NAMES[family_name])

    method_value = params.get("method")
    if method_value is not None:
        method_name = str(method_value).lower()
        if method_name in _METHOD_MARK_NAMES:
            marks.add(_METHOD_MARK_NAMES[method_name])

    case_value = params.get("case")
    if case_value is not None:
        case_fields = _extract_case_like_fields(case_value)
        case_family = _normalize_family_name(case_fields.get("family"))
        if case_family in _FAMILY_MARK_NAMES:
            marks.add(_FAMILY_MARK_NAMES[case_family])
        case_method = case_fields.get("method")
        if case_method is not None:
            case_method_name = str(case_method).lower()
            if case_method_name in _METHOD_MARK_NAMES:
                marks.add(_METHOD_MARK_NAMES[case_method_name])
        if "select" in case_fields:
            marks.add("select_true" if case_fields["select"] else "select_false")

    if "fixed_sp_override" in params and params["fixed_sp_override"] is not None:
        marks.add("method_fixed")

    if "select" in params:
        marks.add("select_true" if params["select"] else "select_false")

    return marks


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath)).name
        marks: set[str] = set()
        marks.update(_DEFAULT_MARKS_BY_FILE.get(path, set()))
        marks.update(_STATUS_MARKS_BY_FILE.get(path, set()))

        if not any(mark.startswith("status_") for mark in marks):
            marks.add("status_stable")

        text_marks = _infer_marks_from_text(_collect_text_fragments(item))
        marks.update(text_marks)
        marks.update(_infer_marks_from_callspec(item))

        if (
            path in _SELECTION_CAPABLE_FILES
            and "select_true" not in marks
            and "select_false" not in marks
        ):
            marks.add("select_false")

        for mark_name in sorted(marks):
            item.add_marker(getattr(pytest.mark, mark_name))
