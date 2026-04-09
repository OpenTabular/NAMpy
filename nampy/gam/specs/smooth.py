from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence, Union


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


@dataclass(frozen=True)
class SumToZeroFactorSmoothSpec(BaseSmoothSpec):
    bs: str = "sz"
    xt: Any = None


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
    special_key = str(special).lower()
    if special_key == "s":
        bs_key = str(bs).lower()
        if bs_key == "cr":
            return CubicRegressionSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                constraint_mode=constraint_mode,
                shared_basis_setup=shared_basis_setup,
                pc=pc,
            )
        if bs_key == "cs":
            return CubicShrinkageSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                constraint_mode=constraint_mode,
                shared_basis_setup=shared_basis_setup,
                pc=pc,
            )
        if bs_key == "cc":
            return CyclicCubicRegressionSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                constraint_mode=constraint_mode,
                shared_basis_setup=shared_basis_setup,
                pc=pc,
            )
        if bs_key == "ps":
            return PSplineSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                m=m,
                constraint_mode=constraint_mode,
                pc=pc,
            )
        if bs_key == "tp":
            return ThinPlateSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                m=m,
                xt=xt,
                constraint_mode=constraint_mode,
                pc=pc,
            )
        if bs_key == "ts":
            return ThinPlateShrinkageSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                m=m,
                xt=xt,
                constraint_mode=constraint_mode,
                pc=pc,
            )
        if bs_key == "gp":
            return GPSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                m=m,
                xt=xt,
                constraint_mode=constraint_mode,
                pc=pc,
            )
        if bs_key == "mrf":
            return MarkovRandomFieldSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                xt=xt,
            )
        if bs_key == "re":
            return RandomEffectSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                xt=xt,
            )
        if bs_key == "fs":
            return FactorSmoothInteractionSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                xt=xt,
            )
        if bs_key == "sz":
            return SumToZeroFactorSmoothSpec(
                special="s",
                k=k,
                fx=fx,
                select=select,
                sp=sp,
                knots=knots,
                xt=xt,
            )
        raise NotImplementedError(f"Unsupported s() basis {bs!r}.")

    if special_key == "te":
        return TensorProductSmoothSpec(
            special="te",
            bs=bs,
            k=k,
            fx=fx,
            select=select,
            sp=sp,
            knots=knots,
        )
    if special_key == "ti":
        return TensorInteractionSmoothSpec(
            special="ti",
            bs=bs,
            k=k,
            fx=fx,
            select=select,
            sp=sp,
            knots=knots,
            mc=mc,
        )
    if special_key == "t2":
        return TensorANOVASmoothSpec(
            special="t2",
            bs=bs,
            k=k,
            fx=fx,
            select=select,
            sp=sp,
            knots=knots,
            full=full,
            ord=ord_,
        )

    raise NotImplementedError(f"Unsupported smooth special {special!r}.")


def smooth_spec_from_basis_options(basis_options: Mapping[str, Any]) -> SmoothSpec:
    opts = dict(basis_options or {})
    return build_smooth_spec(
        special=opts.get("special", "s"),
        bs=opts.get("bs", "cr"),
        k=opts.get("k", -1),
        fx=bool(opts.get("fx", False)),
        select=bool(opts.get("select", False)),
        m=opts.get("m", None),
        xt=opts.get("xt", None),
        sp=opts.get("sp", None),
        pc=opts.get("pc", None),
        knots=opts.get("knots", None),
        constraint_mode=str(opts.get("constraint_mode", "auto")),
        shared_basis_setup=opts.get("shared_basis_setup", None),
        mc=opts.get("mc", None),
        full=bool(opts.get("full", False)),
        ord_=opts.get("ord", None),
    )


def replace_smooth_spec(spec: SmoothSpec, **changes: Any) -> SmoothSpec:
    return replace(spec, **changes)


def smooth_spec_basis_name(spec: SmoothSpec) -> Any:
    return spec.bs


def is_s_type(spec: SmoothSpec) -> bool:
    return spec.special == "s"


def tensor_basis_list(spec: Union[TensorProductSmoothSpec, TensorInteractionSmoothSpec, TensorANOVASmoothSpec], n_features: int) -> list[Any]:
    basis = spec.bs
    if isinstance(basis, str):
        return [basis] * n_features
    if isinstance(basis, Sequence):
        basis_list = list(basis)
        if len(basis_list) != n_features:
            raise ValueError(f"Expected length {n_features}, got {len(basis_list)}.")
        return basis_list
    return [basis] * n_features

