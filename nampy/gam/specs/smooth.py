from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence, Union


@dataclass(frozen=True)
class BaseSmoothSpec:
    special: str
    bs: Any
    k: Any = -1
    fx: bool = False
    select: bool = False
    sp: Any = None
    knots: Any = None

    def to_basis_options(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CubicRegressionSmoothSpec(BaseSmoothSpec):
    bs: str = "cr"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class CyclicCubicRegressionSmoothSpec(BaseSmoothSpec):
    bs: str = "cc"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class CubicShrinkageSmoothSpec(BaseSmoothSpec):
    bs: str = "cs"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class PSplineSmoothSpec(BaseSmoothSpec):
    bs: str = "ps"
    m: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class ThinPlateSmoothSpec(BaseSmoothSpec):
    bs: str = "tp"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class ThinPlateShrinkageSmoothSpec(BaseSmoothSpec):
    bs: str = "ts"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class GPSmoothSpec(BaseSmoothSpec):
    bs: str = "gp"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class MarkovRandomFieldSmoothSpec(BaseSmoothSpec):
    bs: str = "mrf"
    xt: Any = None


@dataclass(frozen=True)
class RandomEffectSmoothSpec(BaseSmoothSpec):
    bs: str = "re"
    xt: Any = None


@dataclass(frozen=True)
class FactorSmoothInteractionSpec(BaseSmoothSpec):
    bs: str = "fs"
    xt: Any = None
    constraint_mode: str = "auto"


@dataclass(frozen=True)
class SumToZeroFactorSmoothSpec(BaseSmoothSpec):
    bs: str = "sz"
    xt: Any = None
    constraint_mode: str = "auto"


@dataclass(frozen=True)
class TensorProductSmoothSpec(BaseSmoothSpec):
    special: str = "te"
    bs: Any = "cr"


@dataclass(frozen=True)
class TensorInteractionSmoothSpec(BaseSmoothSpec):
    special: str = "ti"
    bs: Any = "cr"
    mc: Any = None


@dataclass(frozen=True)
class TensorANOVASmoothSpec(BaseSmoothSpec):
    special: str = "t2"
    bs: Any = "cr"
    full: bool = False
    ord: Any = None


SmoothSpec = Union[
    CubicRegressionSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
    PSplineSmoothSpec,
    ThinPlateSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    GPSmoothSpec,
    MarkovRandomFieldSmoothSpec,
    RandomEffectSmoothSpec,
    FactorSmoothInteractionSpec,
    SumToZeroFactorSmoothSpec,
    TensorProductSmoothSpec,
    TensorInteractionSmoothSpec,
    TensorANOVASmoothSpec,
]

# Merged defaults for dict dispatch (build_smooth_spec / smooth_spec_from_basis_options).
_SMOOTH_SPEC_DEFAULTS: dict[str, Any] = {
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
    "full": False,
    "ord_": None,
}


def _build_s_cr(opts: Mapping[str, Any]) -> CubicRegressionSmoothSpec:
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


def _build_s_cs(opts: Mapping[str, Any]) -> CubicShrinkageSmoothSpec:
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


def _build_s_cc(opts: Mapping[str, Any]) -> CyclicCubicRegressionSmoothSpec:
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


def _build_s_ps(opts: Mapping[str, Any]) -> PSplineSmoothSpec:
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


def _build_s_tp(opts: Mapping[str, Any]) -> ThinPlateSmoothSpec:
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


def _build_s_ts(opts: Mapping[str, Any]) -> ThinPlateShrinkageSmoothSpec:
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


def _build_s_gp(opts: Mapping[str, Any]) -> GPSmoothSpec:
    return GPSmoothSpec(
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


def _build_s_mrf(opts: Mapping[str, Any]) -> MarkovRandomFieldSmoothSpec:
    return MarkovRandomFieldSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_re(opts: Mapping[str, Any]) -> RandomEffectSmoothSpec:
    return RandomEffectSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_fs(opts: Mapping[str, Any]) -> FactorSmoothInteractionSpec:
    return FactorSmoothInteractionSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


def _build_s_sz(opts: Mapping[str, Any]) -> SumToZeroFactorSmoothSpec:
    return SumToZeroFactorSmoothSpec(
        special="s",
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        xt=opts["xt"],
    )


# New s() basis: add ``_build_s_<bs>(opts)`` and one registry line below.
_S_BASIS_SPEC_BUILDERS: dict[str, Callable[[Mapping[str, Any]], BaseSmoothSpec]] = {
    "cr": _build_s_cr,
    "cs": _build_s_cs,
    "cc": _build_s_cc,
    "ps": _build_s_ps,
    "tp": _build_s_tp,
    "ts": _build_s_ts,
    "gp": _build_s_gp,
    "mrf": _build_s_mrf,
    "re": _build_s_re,
    "fs": _build_s_fs,
    "sz": _build_s_sz,
}


def _build_te(opts: Mapping[str, Any]) -> TensorProductSmoothSpec:
    return TensorProductSmoothSpec(
        special="te",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
    )


def _build_ti(opts: Mapping[str, Any]) -> TensorInteractionSmoothSpec:
    return TensorInteractionSmoothSpec(
        special="ti",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        mc=opts["mc"],
    )


def _build_t2(opts: Mapping[str, Any]) -> TensorANOVASmoothSpec:
    return TensorANOVASmoothSpec(
        special="t2",
        bs=opts["bs"],
        k=opts["k"],
        fx=opts["fx"],
        select=opts["select"],
        sp=opts["sp"],
        knots=opts["knots"],
        full=opts["full"],
        ord=opts["ord_"],
    )


# New tensor special (te/ti/t2-style): add ``_build_<name>(opts)`` and one line here.
_SPECIAL_SMOOTH_BUILDERS: dict[str, Callable[[Mapping[str, Any]], SmoothSpec]] = {
    "te": _build_te,
    "ti": _build_ti,
    "t2": _build_t2,
}


def _dispatch_smooth_spec_from_options(opts: Mapping[str, Any]) -> SmoothSpec:
    merged = {**_SMOOTH_SPEC_DEFAULTS, **dict(opts)}
    special_key = str(merged["special"]).lower()
    if special_key == "s":
        bs_key = str(merged["bs"]).lower()
        builder = _S_BASIS_SPEC_BUILDERS.get(bs_key)
        if builder is None:
            raise NotImplementedError(f"Unsupported s() basis {merged['bs']!r}.")
        return builder(merged)
    builder = _SPECIAL_SMOOTH_BUILDERS.get(special_key)
    if builder is None:
        raise NotImplementedError(f"Unsupported smooth special {merged['special']!r}.")
    return builder(merged)


def build_smooth_spec(
    *,
    special: str,
    bs: Any,
    k: Any = -1,
    fx: bool = False,
    select: bool = False,
    m: Any = None,
    xt: Any = None,
    sp: Any = None,
    pc: Any = None,
    knots: Any = None,
    constraint_mode: str = "auto",
    shared_basis_setup: Any = None,
    mc: Any = None,
    full: bool = False,
    ord_: Any = None,
) -> SmoothSpec:
    return _dispatch_smooth_spec_from_options(locals())


def smooth_spec_from_basis_options(basis_options: Mapping[str, Any]) -> SmoothSpec:
    raw = dict(basis_options or {})
    if "ord_" not in raw and "ord" in raw:
        raw = {**raw, "ord_": raw["ord"]}
    merged = {**_SMOOTH_SPEC_DEFAULTS, **raw}
    merged["fx"] = bool(merged.get("fx", False))
    merged["select"] = bool(merged.get("select", False))
    merged["full"] = bool(merged.get("full", False))
    merged["constraint_mode"] = str(merged.get("constraint_mode", "auto"))
    return _dispatch_smooth_spec_from_options(merged)


def replace_smooth_spec(spec: SmoothSpec, **changes: Any) -> SmoothSpec:
    return replace(spec, **changes)


def smooth_spec_basis_name(spec: SmoothSpec) -> Any:
    return spec.bs


def is_s_type(spec: SmoothSpec) -> bool:
    return spec.special == "s"


def tensor_basis_list(
    spec: Union[
        TensorProductSmoothSpec, TensorInteractionSmoothSpec, TensorANOVASmoothSpec
    ],
    n_features: int,
) -> list[Any]:
    basis = spec.bs
    if isinstance(basis, str):
        return [basis] * n_features
    if isinstance(basis, Sequence):
        basis_list = list(basis)
        if len(basis_list) != n_features:
            raise ValueError(f"Expected length {n_features}, got {len(basis_list)}.")
        return basis_list
    return [basis] * n_features
