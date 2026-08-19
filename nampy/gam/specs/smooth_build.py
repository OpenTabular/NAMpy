"""Construction of canonical smooth-term specifications from options."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from .smooth import (
    CubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    FactorSmoothInteractionSpec,
    PSplineSmoothSpec,
    RandomEffectSmoothSpec,
    SmoothSpec,
    SumToZeroFactorSmoothSpec,
    TensorInteractionSmoothSpec,
    TensorProductSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    ThinPlateSmoothSpec,
)

_SMOOTH_SPEC_DEFAULTS: dict[str, object] = {
    "special": "s",
    "bs": "cr",
    "k": -1,
    "fx": False,
    "select": False,
    "m": None,
    "xt": None,
    "sp": None,
    "pc": None,
    "knots": None,
    "constraint_mode": "auto",
    "shared_basis_setup": None,
    "mc": None,
    "d": None,
}

def _build_s_cr(opts) -> CubicRegressionSmoothSpec:
    return CubicRegressionSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_cs(opts) -> CubicShrinkageSmoothSpec:
    return CubicShrinkageSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_cc(opts) -> CyclicCubicRegressionSmoothSpec:
    return CyclicCubicRegressionSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        constraint_mode=opts["constraint_mode"],
        shared_basis_setup=opts["shared_basis_setup"],
        pc=opts["pc"],
    )


def _build_s_ps(opts) -> PSplineSmoothSpec:
    return PSplineSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_tp(opts) -> ThinPlateSmoothSpec:
    return ThinPlateSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        xt=opts["xt"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_ts(opts) -> ThinPlateShrinkageSmoothSpec:
    return ThinPlateShrinkageSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        m=opts["m"],
        xt=opts["xt"],
        constraint_mode=opts["constraint_mode"],
        pc=opts["pc"],
    )


def _build_s_re(opts) -> RandomEffectSmoothSpec:
    return RandomEffectSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_fs(opts) -> FactorSmoothInteractionSpec:
    return FactorSmoothInteractionSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_sz(opts) -> SumToZeroFactorSmoothSpec:
    return SumToZeroFactorSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


_S_BASIS_SPEC_BUILDERS: dict[str, Callable[[dict[str, Any]], SmoothSpec]] = {
    "cr": _build_s_cr,
    "cs": _build_s_cs,
    "cc": _build_s_cc,
    "ps": _build_s_ps,
    "tp": _build_s_tp,
    "ts": _build_s_ts,
    "re": _build_s_re,
    "fs": _build_s_fs,
    "sz": _build_s_sz,
}


def _build_te(opts) -> TensorProductSmoothSpec:
    return TensorProductSmoothSpec(
        special="te",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        m=opts["m"],
        xt=opts["xt"],
        sp=opts["sp"],
        knots=opts["knots"],
        d=opts["d"],
    )


def _build_ti(opts) -> TensorInteractionSmoothSpec:
    return TensorInteractionSmoothSpec(
        special="ti",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        m=opts["m"],
        xt=opts["xt"],
        sp=opts["sp"],
        knots=opts["knots"],
        mc=opts["mc"],
        d=opts["d"],
    )


_SPECIAL_SMOOTH_BUILDERS: dict[str, Callable[[dict[str, Any]], SmoothSpec]] = {
    "te": _build_te,
    "ti": _build_ti,
}


def _is_vector_fx(fx) -> bool:
    return fx is not None and not np.isscalar(fx)


_PC_SUPPORTED_S_BASES = {"cc", "cr", "cs", "ps", "tp", "ts"}


def _dispatch_smooth_spec_from_options(opts) -> SmoothSpec:
    merged = {**_SMOOTH_SPEC_DEFAULTS, **dict(opts)}
    special_key = str(merged["special"]).lower()
    has_pc = merged.get("pc") is not None
    if special_key == "s":
        bs_key = str(merged["bs"]).lower()
        builder = _S_BASIS_SPEC_BUILDERS.get(bs_key)
        if builder is None:
            raise NotImplementedError(f"Unsupported s() basis {merged['bs']!r}.")
        if has_pc and bs_key not in _PC_SUPPORTED_S_BASES:
            raise NotImplementedError(
                f"pc= is not supported for s(..., bs={merged['bs']!r}); "
                "point constraints are only supported for bs in "
                "{'cc', 'cr', 'cs', 'ps', 'tp', 'ts'}."
            )
        return builder(merged)
    if has_pc:
        raise NotImplementedError(
            f"pc= is not supported for {special_key}(...) smooths."
        )
    builder = _SPECIAL_SMOOTH_BUILDERS.get(special_key)
    if builder is None:
        raise NotImplementedError(f"Unsupported smooth special {merged['special']!r}.")
    return builder(merged)


def build_smooth_spec(
    *,
    special: str,
    bs,
    k=-1,
    fx=False,
    select: bool = False,
    m=None,
    xt=None,
    sp=None,
    pc=None,
    knots=None,
    constraint_mode: str = "auto",
    shared_basis_setup=None,
    mc=None,
    d=None,
) -> SmoothSpec:
    return _dispatch_smooth_spec_from_options(locals())


def _flatten_formula_arg(value):
    if isinstance(value, np.ndarray):
        return list(np.asarray(value, dtype=object).ravel())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _round_formula_arg(value):
    vals = _flatten_formula_arg(value)
    rounded = [int(np.rint(float(v))) for v in vals]
    if len(rounded) == 1 and not isinstance(value, (list, tuple, np.ndarray)):
        return rounded[0]
    return rounded


def _normalize_univariate_smooth_k(k):
    # mgcv/R/smooth.r::s(): k.new <- round(k), with warning on change.
    rounded = _round_formula_arg(k)
    old_vals = [float(v) for v in _flatten_formula_arg(k)]
    new_vals = [float(v) for v in _flatten_formula_arg(rounded)]
    if old_vals != new_vals:
        warnings.warn(
            "argument k of s() should be integer and has been rounded",
            stacklevel=3,
        )
    return rounded


def _normalize_tensor_dimensions(d, n_features):
    if d is None:
        return [1] * int(n_features)
    vals = _flatten_formula_arg(d)
    if any(pd.isna(v) for v in vals):
        return [1] * int(n_features)
    rounded = [int(np.rint(float(v))) for v in vals]
    ok = bool(rounded) and all(v > 0 for v in rounded)
    ok = ok and sum(rounded) == int(n_features)
    if not ok:
        warnings.warn("something wrong with argument d.", stacklevel=3)
        return [1] * int(n_features)
    return rounded


def _normalize_tensor_k(k, d):
    # mgcv/R/smooth.r::te(): invalid tensor k resets to 5^d (d = 1 here).
    d = [int(v) for v in d]
    n_bases = len(d)
    if k is None:
        return [int(5**di) for di in d]
    vals = _flatten_formula_arg(k)
    if any(pd.isna(v) for v in vals):
        return [int(5**di) for di in d]
    rounded = [int(np.rint(float(v))) for v in vals]
    ok = True
    if any(v < 3 for v in rounded):
        ok = False
        warnings.warn(
            "one or more supplied k too small - reset to default",
            stacklevel=3,
        )
    if len(rounded) == 1 and ok:
        return rounded * int(n_bases)
    if len(rounded) != int(n_bases):
        ok = False
    if not ok:
        return [int(5**di) for di in d]
    return rounded


def _normalize_tensor_basis(bs, d):
    d = [int(v) for v in d]
    if isinstance(bs, str):
        out = [str(bs)] * len(d)
    else:
        out = [str(v) for v in _flatten_formula_arg(bs)]
    if len(out) != len(d):
        warnings.warn("bs wrong length and ignored.", stacklevel=3)
        out = ["cr"] * len(d)
    return [
        "tp" if di > 1 and b in {"cr", "cs", "ps", "cp"} else b
        for b, di in zip(out, d, strict=True)
    ]


def _normalize_smooth_id(smoothing_id):
    # mgcv/R/smooth.r::s()/te(): only first element of multi-element id used.
    if smoothing_id is None:
        return None
    if isinstance(smoothing_id, str):
        return str(smoothing_id)
    vals = _flatten_formula_arg(smoothing_id)
    if len(vals) > 1:
        warnings.warn("only first element of `id' used", stacklevel=3)
        vals = vals[:1]
    if len(vals) == 1:
        return str(vals[0])
    return str(smoothing_id)


def _tensor_feature_groups(features, d):
    features = [str(f) for f in features]
    d = [1] * len(features) if d is None else [int(v) for v in d]
    groups = []
    pos = 0
    for di in d:
        groups.append(tuple(features[pos : pos + di]))
        pos += di
    return groups


def smooth_spec_from_basis_options(basis_options) -> SmoothSpec:
    raw = dict(basis_options or {})
    merged = {**_SMOOTH_SPEC_DEFAULTS, **raw}
    special_key = str(merged.get("special", "s")).lower()
    fx_raw = merged.get("fx", False)
    if _is_vector_fx(fx_raw):
        if special_key in {"te", "ti"}:
            merged["fx"] = [bool(value) for value in _flatten_formula_arg(fx_raw)]
        else:
            raise NotImplementedError(
                "Vector-valued fx is not supported; use a scalar boolean."
            )
    else:
        merged["fx"] = bool(fx_raw)
    merged["select"] = bool(merged.get("select", False))
    merged["constraint_mode"] = str(merged.get("constraint_mode", "auto"))
    return _dispatch_smooth_spec_from_options(merged)


def _coerce_fx(fx, *, kind, n_features):
    kind_key = str(kind).lower()
    if kind_key in {"te", "ti"} and isinstance(fx, (list, tuple, np.ndarray)):
        values = [bool(value) for value in _flatten_formula_arg(fx)]
        if len(values) != int(n_features):
            warnings.warn("dimension of fx is wrong", stacklevel=3)
            return [False] * int(n_features)
        return values
    if isinstance(fx, (list, tuple, np.ndarray)):
        raise NotImplementedError(
            "Vector-valued fx is not supported; use a scalar boolean."
        )
    return bool(fx)


def _default_k_for_basis(_basis, default_k):
    return default_k


def _factor_smooth_base_basis_from_xt(xt):
    if xt is None:
        return "tp"
    if isinstance(xt, str):
        return str(xt).lower()
    if isinstance(xt, dict):
        return str(xt.get("bs", "tp")).lower()
    return None


def _default_k_for_smooth(kind, basis, features, default_k):
    kind_key = str(kind).lower()
    if kind_key in {"te", "ti"}:
        # mgcv::te()/ti() default k to 5^d per marginal. The current
        # Python tensor surface supports one feature per marginal, so d = 1.
        return [5] * len(features)
    if str(basis).lower() in {"tp", "ts"}:
        # mgcv/R/smooth.r::s() leaves k = -1; smooth.construct.tp.smooth.spec
        # resolves the d-dependent default M + c(8, 27, 100)[min(d, 3)] at
        # construction time (mgcv/R/smooth.r:1316-1318). A flat default here
        # would wrongly give k = 10 for d > 1.
        return -1
    return _default_k_for_basis(basis, default_k)


def _default_basis_for_kind(kind, default_basis):
    kind_key = str(kind).lower()
    if kind_key == "s":
        return "tp" if default_basis is None else default_basis
    if default_basis is not None and isinstance(default_basis, (list, tuple)):
        return default_basis
    return "cr"


def _knots_for_features(knots, features):
    if knots is None:
        return None
    if isinstance(knots, dict):
        vals = [knots.get(str(f), None) for f in features]
        return (
            None
            if all(v is None for v in vals)
            else vals[0] if len(features) == 1 else vals
        )
    return knots



__all__ = ["build_smooth_spec", "smooth_spec_from_basis_options"]
