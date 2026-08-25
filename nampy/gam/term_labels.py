"""Canonical mgcv-style labels for smooth and parametric terms."""

from __future__ import annotations

import re


def normalize_mgcv_term_label(label):
    """Drop constructor options that upstream ``predict.gam`` omits from labels."""
    if label is None:
        return None
    text = str(label)
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if 0 <= open_idx < close_idx:
        fn = text[:open_idx].strip()
        inner = text[open_idx + 1 : close_idx]
        suffix = text[close_idx + 1 :].strip()
        suffix_has_factor_level = re.match(r"^:\s*[^=:\s]+\s*=", suffix) is not None
        args = []
        current = []
        depth = 0
        quote = None
        escape = False
        for ch in inner:
            if quote is not None:
                current.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                current.append(ch)
                continue
            if ch in "([{":
                depth += 1
                current.append(ch)
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                current.append(ch)
                continue
            if ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    args.append(part)
                current = []
                continue
            current.append(ch)
        part = "".join(current).strip()
        if part:
            args.append(part)

        kept = []
        for arg in args:
            name = arg.split("=", 1)[0].strip().lower() if "=" in arg else ""
            if name in {"bs", "k", "m", "sp", "id", "mc", "fx", "xt"}:
                continue
            if suffix_has_factor_level and name == "by":
                continue
            kept.append(arg)
        text = f"{fn}({', '.join(kept)}){suffix}"
    else:
        text = re.sub(r",\s*bs\s*=\s*(\"[^\"]*\"|'[^']*'|[^,)]+)", "", text)
        text = re.sub(r",\s*k\s*=\s*[^,)]+", "", text)
        text = re.sub(r",\s*m\s*=\s*[^,)]+", "", text)
        text = re.sub(r",\s*sp\s*=\s*[^,)]+", "", text)
        text = re.sub(r",\s*id\s*=\s*(\"[^\"]*\"|'[^']*'|[^,)]+)", "", text)
        text = re.sub(r",\s*xt\s*=\s*(\"[^\"]*\"|'[^']*'|[^,)]+)", "", text)
    text = re.sub(
        r"^([a-zA-Z0-9_]+\([^)]*?)(?:,\s*by\s*=\s*([^)]+))\)$",
        lambda match: f"{match.group(1)}):{match.group(2).strip()}",
        text,
    )
    text = re.sub(
        r":\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*=\s*([^:\s]+)$",
        lambda match: f":{match.group(1)}{match.group(2)}",
        text,
    )
    return text


def multi_predictor_term_label(
    label: str, *, predictor_index: int, term_type: str
) -> str:
    """Apply mgcv's formula-list suffix to later-predictor term labels."""
    if int(predictor_index) <= 0:
        return str(label)
    if str(term_type) == "parametric":
        return f"{label}.{int(predictor_index)}"
    open_index = str(label).find("(")
    if open_index < 0:
        return f"{label}.{int(predictor_index)}"
    return f"{label[:open_index]}.{int(predictor_index)}{label[open_index:]}"


def mgcv_term_display_label(term, *, formula_parametric: bool = False) -> str:
    """Return a compiled term's predictor-aware public mgcv label."""
    term_type = str(getattr(term, "term_type", ""))
    label = str(getattr(term, "label", ""))
    if term_type == "parametric" and formula_parametric:
        metadata = dict(getattr(term, "metadata", {}) or {})
        label = str(metadata.get("formula_term", label))
    elif term_type != "parametric":
        label = str(normalize_mgcv_term_label(label))
    return multi_predictor_term_label(
        label,
        predictor_index=int(getattr(term, "predictor_index", 0)),
        term_type=term_type,
    )


def compiled_term_display_label(term) -> str:
    """Return the predictor-aware label retained by the compiled NAMpy term.

    Unlike :func:`mgcv_term_display_label`, this deliberately keeps constructor
    arguments.  Some diagnostics expose the compiled formula identity rather
    than the compact column names returned by ``predict.gam(type="terms")``.
    """
    return multi_predictor_term_label(
        str(getattr(term, "label", "")),
        predictor_index=int(getattr(term, "predictor_index", 0)),
        term_type=str(getattr(term, "term_type", "")),
    )


__all__ = [
    "compiled_term_display_label",
    "mgcv_term_display_label",
    "multi_predictor_term_label",
    "normalize_mgcv_term_label",
]
